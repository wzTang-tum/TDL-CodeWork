import numpy as np
import os 
import pickle as pkl
# from emnist import extract_training_samples,extract_test_samples
import numpy as np
import struct
from tensorflow.keras.utils import to_categorical
import json
import gzip

def load_mnist_data(mode):
    if mode == 'train':
        file_path = '../Data/MNIST//train-images.idx3-ubyte'
        label_path = '../Data/MNIST//train-labels.idx1-ubyte'
    else:
        file_path = '../Data/MNIST//t10k-images.idx3-ubyte'
        label_path = '../Data/MNIST//t10k-labels.idx1-ubyte'
        
    binfile = open(file_path, 'rb') 
    buffers = binfile.read()
    magic,num,rows,cols = struct.unpack_from('>IIII',buffers, 0)
    bits = num * rows * cols
    images = struct.unpack_from('>' + str(bits) + 'B', buffers, struct.calcsize('>IIII'))
    binfile.close()
    images = np.reshape(images, [num, rows * cols])
    
    images = images.reshape((len(images),28,28,1))
    
    binfile = open(label_path, 'rb')
    buffers = binfile.read()
    magic,num = struct.unpack_from('>II', buffers, 0) 
    labels = struct.unpack_from('>' + str(num) + "B", buffers, struct.calcsize('>II'))
    binfile.close()
    labels = np.reshape(labels, [num])
    
    images = images/255
    labels = to_categorical(labels)
    
    return images,labels


def load_femnist_data():
    # Paths relative to Code/ directory
    train_images_path = '../Data/FEMNIST/emnist-byclass-train-images-idx3-ubyte.gz'
    train_labels_path = '../Data/FEMNIST/emnist-byclass-train-labels-idx1-ubyte.gz'
    test_images_path = '../Data/FEMNIST/emnist-byclass-test-images-idx3-ubyte.gz'
    test_labels_path = '../Data/FEMNIST/emnist-byclass-test-labels-idx1-ubyte.gz'

    def read_images(path):
        with gzip.open(path, 'rb') as f:
            buffers = f.read()
            magic, num, rows, cols = struct.unpack_from('>IIII', buffers, 0)
            bits = num * rows * cols
            images = struct.unpack_from('>' + str(bits) + 'B', buffers, struct.calcsize('>IIII'))
            images = np.reshape(images, [num, rows * cols])
            images = images.reshape((len(images), 28, 28, 1))
            return images

    def read_labels(path):
        with gzip.open(path, 'rb') as f:
            buffers = f.read()
            magic, num = struct.unpack_from('>II', buffers, 0)
            labels = struct.unpack_from('>' + str(num) + "B", buffers, struct.calcsize('>II'))
            labels = np.reshape(labels, [num])
            return labels

    train_images = read_images(train_images_path)
    train_labels = read_labels(train_labels_path)
    test_images = read_images(test_images_path)
    test_labels = read_labels(test_labels_path)

    train_images = train_images/255
    test_images = test_images/255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    
    return train_images, train_labels, test_images, test_labels


def load_cifar10_data(path='../Data/CIFAR10'):
    
    train_data = []
    train_label = []
    for i in range(5):
        with open(os.path.join(path,'data_batch_'+str(i+1)),'rb') as f:
            data = pkl.load(f, encoding='bytes')
            train_data.append(data[b'data'])
            train_label.append(data[b'labels'])
    train_data = np.concatenate(train_data,axis=0)
    train_label = np.concatenate(train_label,axis=0)
    train_data = train_data.reshape((-1,3,32,32))
    
    with open(os.path.join(path,'test_batch'),'rb') as f:
        data = pkl.load(f, encoding='bytes')
        test_data = data[b'data'].reshape((-1,3,32,32))
        test_label = data[b'labels']
        
    test_label = np.array(test_label)
        
    train_images = train_data/255
    test_images = test_data/255
    

    train_labels = to_categorical(train_label)
    test_labels = to_categorical(test_label)

    train_images = np.transpose(train_images,(0,2,3,1))
    test_images  = np.transpose(test_images, (0,2,3,1))

    return train_images,train_labels,test_images,test_labels

def load_data(dataset):
    
    if dataset == 'MNIST':
        train_images, train_labels = load_mnist_data('train')
        test_images, test_labels = load_mnist_data('test')

    elif dataset == 'FEMNIST':
        train_images, train_labels, test_images, test_labels = load_femnist_data()
    elif dataset == 'CIFAR10':
        train_images,train_labels, test_images,test_labels = load_cifar10_data()

    return train_images, train_labels, test_images, test_labels, train_images.shape[-1],train_labels.shape[-1]

def client_partation(train_labels,range_length = 1000):
    index = np.random.permutation(len(train_labels))

    train_users = {}

    for i in range(int(np.ceil(len(train_labels)//range_length))):
        s,ed = range_length*i, (i+1)*range_length
        ed = min(ed,len(train_labels))
        train_users[i] = index[s:ed]
        
    return train_users


def client_partition_mild_noniid(labels, num_clients, extra_class_fraction=0.1):
    """
    Mild non-IID partition for FL.
    
    Parameters:
    - labels: ndarray, shape (num_samples,) or one-hot
    - num_clients: int
    - extra_class_fraction: float, fraction of samples to bias for extra classes
    """
    if labels.ndim == 2:  # one-hot
        labels_int = np.argmax(labels, axis=1)
    else:
        labels_int = labels

    num_samples = len(labels_int)
    samples_per_client = num_samples // num_clients

    train_users = {}
    all_indices = np.random.permutation(num_samples)

    for client_id in range(num_clients):
        start = client_id * samples_per_client
        end = start + samples_per_client
        client_indices = all_indices[start:end].tolist()
        
        # Mild non-IID: 增加 fraction 的偏向类别样本
        bias_size = int(len(client_indices) * extra_class_fraction)
        if bias_size > 0:
            # 随机选一个偏向类别
            client_labels = labels_int[client_indices]
            cls_counts = np.bincount(client_labels)
            preferred_class = np.argmax(cls_counts)
            
            # 找一些同类样本替换
            same_class_indices = np.where(labels_int == preferred_class)[0]
            same_class_indices = np.setdiff1d(same_class_indices, client_indices)
            n_replace = min(bias_size, len(same_class_indices))
            client_indices[:n_replace] = same_class_indices[:n_replace]
        
        train_users[client_id] = client_indices
    
    return train_users




def dump_result(Res,dataset,attack_mode,defense_mode,taxic_ratio,alpha,epsilon):
    Key = '-'.join([dataset,attack_mode,defense_mode,str(taxic_ratio),str(alpha),str(epsilon)])
    with open('../Result/'+Key+'.json','a') as f:
        s = json.dumps(Res) + '\n'
        f.write(s)