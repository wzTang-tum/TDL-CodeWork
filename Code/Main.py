import os

import tensorflow as tf

# Use TF1 compatibility when available so code works with TF2 installs
tf.compat.v1.disable_eager_execution()

 

from preprocessing import *
from poison import *
from FLTrain import *
from model import *
from sklearn.metrics import f1_score

import click


@click.command()
@click.option('-d', '--dataset',default='MNIST')
@click.option('-a', '--attack-mode', default='AttackDPFL')
@click.option('-m', '--defense-mode', default='RobustDPFL')
@click.option('-t', '--taxic-ratio', default=0.1)
@click.option('-o', '--alpha', default=1.2)
@click.option('-e', '--epsilon', default=5)
@click.option('-g', '--gpu', default=0)
def main(dataset,attack_mode,defense_mode,taxic_ratio,alpha,epsilon,gpu):

    # If gpu is negative or empty, do not set CUDA_VISIBLE_DEVICES (use CPU)
    try:
        gpu_int = int(gpu)
    except Exception:
        gpu_int = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_int)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)

    # Use TF1 compat API to set Keras session (avoids deprecation warning)
    tf.compat.v1.keras.backend.set_session(session)

    local_data_size = {'MNIST':400,'FEMNIST':400,'CIFAR10':1000}[dataset]
    EPOCH = {'MNIST':50,'FEMNIST':100,'CIFAR10':50}[dataset]

    train_images, train_labels, test_images, test_labels, NUM_CHANNEL, NUM_CLASS = load_data(dataset)
    train_users = client_partation(train_labels,local_data_size)

    trigger = TriggerGenerator(3,25,25,255,0,NUM_CHANNEL,NUM_CLASS)
    train_images, train_labels, taxic_clients = posison_data(taxic_ratio,trigger,train_users,train_images,train_labels)
    trigger_test_images, trigger_test_labels = build_posison_eval_data(trigger,test_images,test_labels)

    lr = 0.01
    user_num = 100
    ckpt = 5

    norm = 3
    sigma = computing_sigma(alpha,epsilon,norm)

    model = get_model(dataset,lr,train_images.shape[1],NUM_CHANNEL,NUM_CLASS)
    Res = []
    for i in range(EPOCH):
        FL(attack_mode,defense_mode,user_num,model,taxic_clients,train_users,train_images,train_labels,sigma,norm, round_idx=i, total_epochs=EPOCH)

        if i%ckpt==0:
            pred = model.predict(test_images).argmax(axis=-1)
            lbs = test_labels.argmax(axis=-1)
            micro_f1 = f1_score(lbs,pred,average='micro')
            macro_f1 = f1_score(lbs,pred,average='macro')
            
            pred = model.predict(trigger_test_images).argmax(axis=-1)
            lbs = trigger_test_labels.argmax(axis=-1)
            trigger_micro_f1 = f1_score(lbs,pred,average='micro')

            Res.append([micro_f1,trigger_micro_f1,macro_f1])

            print(i,micro_f1,trigger_micro_f1)
            
    dump_result(Res,dataset,attack_mode,defense_mode,taxic_ratio,alpha,epsilon)

if __name__ == '__main__':
    main()