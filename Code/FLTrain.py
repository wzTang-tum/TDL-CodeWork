import numpy as np
from sklearn.cluster import KMeans

# ========== NEW CODE: Gradient Collection ==========
import os
import json
# ========== END NEW CODE ==========

def computing_sigma(alpha,gamma,norm):
    delta = 10**(-8)
    epsilon = gamma - np.log(delta)/(alpha-1)
    sigma = np.sqrt(2*np.log(1.25/delta))*norm/epsilon
    return sigma
    
def LDP(weights,sigma,delta):
    norm = np.sqrt(np.square(weights).sum())
    if norm > delta and sigma>0:
        weights = weights*delta/norm
    if sigma>0:
        weights = weights + np.random.normal(0,sigma,size=weights.shape)
    return weights

def renorm(w1,w2):
    n1 = np.sqrt(np.square(w1).sum())
    n2 = np.sqrt(np.square(w2).sum())
    return w1*n2/n1

def func_flatten_weights(weights):
    a = []
    for i in range(len(weights)):
        w = weights[i].reshape((-1,))
        a.append(w)
    a = np.concatenate(a)
    return a 

def func_unflatten_weights(f_weights,old_weights):
    r_weights = []
    start = 0
    for i in range(len(old_weights)):
        ed = start + old_weights[i].reshape((-1,)).shape[0]
        lw = f_weights[start:ed]
        lw = lw.reshape(old_weights[i].shape)
        r_weights.append(lw)
        start = ed
    return r_weights

def RobustAggergation(mode,all_weights,old_weights):
    flaten_all_weights = all_weights
    
    if mode == 'FedAvg':
        agg_weights = np.mean(flaten_all_weights,axis=0)
    
    elif mode =='RobustDPFL':
        z = np.abs(flaten_all_weights.mean(axis=-1))
        
        # Handle NaN and inf values by replacing with median of finite values
        if np.any(np.isnan(z)) or np.any(np.isinf(z)):
            nan_count = np.isnan(z).sum()
            inf_count = np.isinf(z).sum()
            print(f"[INFO] Detected {nan_count} NaN and {inf_count} inf values in z, replacing with median...")
            
            # Find finite values and compute median
            finite_mask = np.isfinite(z)
            if np.any(finite_mask):
                median_val = np.median(z[finite_mask])
                z[~finite_mask] = median_val
                print(f"[INFO] Replaced abnormal values with median={median_val:.6f}")
            else:
                # Fallback: if all values are NaN/inf, use mean of absolute original values
                # This preserves some signal from the original computation
                z = np.abs(z)
                z[~np.isfinite(z)] = 1.0  # Use neutral value for clustering
                print(f"[INFO] All z values were NaN/inf, using fallback value 1.0")
        
        y_pred = KMeans(n_clusters=2, random_state=9).fit_predict(z.reshape((-1,1)))
        index0 = np.where(y_pred==0)[0]
        index1 = np.where(y_pred==1)[0]
        if z[index0].mean()<z[index1].mean():
            index = index0
        else:
            index = index1
        agg_weights = flaten_all_weights[index].mean(axis=0)
                
    agg_weights = func_unflatten_weights(agg_weights,old_weights)
    
    return agg_weights


def FL(attack_mode,mode,user_num,model,taxic_clients,train_users,train_images,train_labels,sigma=0.1,delta=10,intre_epoch=5, round_idx=None, total_epochs=None):

    user_indexs = np.random.permutation(len(train_users))[:user_num]
    
    old_weights = model.get_weights()
    flatten_old_weights = func_flatten_weights(old_weights)
    all_weights = []
    
    # ========== NEW CODE: Initialize gradient collection for last round ==========
    is_last_round = (round_idx is not None) and (total_epochs is not None) and (round_idx == total_epochs - 1)
    collected_gradients = {
        "malicious": [],
        "benign": [],
        "selected": None
    }
    # ========== END NEW CODE ==========
    
    f_possioned_gradients = []
    quotas = []
    for ui in range(len(user_indexs)):
        ui = user_indexs[ui]
        sample_indexs = train_users[ui]
        x = train_images[sample_indexs]
        y = train_labels[sample_indexs]
        bz = len(x)//5
        for i in range(intre_epoch):
            model.fit(x,y,verbose=0)
        
        weights = model.get_weights()
        weights = func_flatten_weights(weights)
        
        delta_weights = weights-flatten_old_weights
        
        # ========== NEW CODE: Collect gradient if last round ==========
        if is_last_round:
            if ui in taxic_clients:
                collected_gradients["malicious"].append(delta_weights.copy())
            else:
                collected_gradients["benign"].append(delta_weights.copy())
        # ========== END NEW CODE ==========
        
        if ui in taxic_clients:
            if attack_mode == 'AttackNaive':
                delta_weights = LDP(delta_weights,sigma,delta)
                weights = delta_weights+flatten_old_weights
                all_weights.append(weights)
            elif attack_mode == 'AttackNonDP':
                weights = delta_weights+flatten_old_weights
                all_weights.append(weights)
            elif attack_mode == 'AttackDPFL':
                weights2 = LDP(delta_weights,sigma,delta)
                weights2 = weights2+flatten_old_weights
                weights = delta_weights+flatten_old_weights
                weights = renorm(weights,weights2)
                all_weights.append(weights)
        else:
            delta_weights = LDP(delta_weights,sigma,delta)
            weights = delta_weights+flatten_old_weights
            all_weights.append(weights)
            
        model.set_weights(old_weights)

    all_weights = np.array(all_weights)
    weights = RobustAggergation(mode,all_weights,old_weights)    
    
    # ========== NEW CODE: Collect selected gradients from aggregation ==========
    if is_last_round:
        # Extract indices of selected gradients from aggregation
        # (RobustDPFL uses KMeans clustering; we extract the selected cluster)
        # Note: For simplicity, we're storing the aggregated gradient as "selected"
        
        # Convert to lists for JSON serialization - trim large gradients for visualization
        MAX_GRADIENT_SIZE = 10000  # Keep first 10k elements per gradient for plotting
        
        json_data = {}
        
        # Process malicious gradients
        if collected_gradients["malicious"]:
            malicious_trimmed = []
            for g in collected_gradients["malicious"]:
                grad_list = g.tolist()
                if len(grad_list) > MAX_GRADIENT_SIZE:
                    grad_list = grad_list[:MAX_GRADIENT_SIZE]
                malicious_trimmed.append(grad_list)
            json_data["malicious"] = malicious_trimmed
            json_data["malicious_count"] = len(collected_gradients["malicious"])
        
        # Process benign gradients
        if collected_gradients["benign"]:
            benign_trimmed = []
            for g in collected_gradients["benign"]:
                grad_list = g.tolist()
                if len(grad_list) > MAX_GRADIENT_SIZE:
                    grad_list = grad_list[:MAX_GRADIENT_SIZE]
                benign_trimmed.append(grad_list)
            json_data["benign"] = benign_trimmed
            json_data["benign_count"] = len(collected_gradients["benign"])
        
        # Process selected gradient
        selected_list = [g.tolist() if hasattr(g, 'tolist') else g for g in weights]
        selected_trimmed = []
        for g in selected_list:
            if isinstance(g, list) and len(g) > MAX_GRADIENT_SIZE:
                g = g[:MAX_GRADIENT_SIZE]
            selected_trimmed.append(g)
        json_data["selected"] = selected_trimmed
        
        # Create output directory if not exists
        output_dir = "../Result/gradient_analysis"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSON format
        output_json = os.path.join(output_dir, "gradients_last_round.json")
        
        with open(output_json, 'w') as f:
            json.dump(json_data, f)
        
        print(f"[GRADIENT COLLECTION] Saved gradients to {output_json}")
        print(f"[GRADIENT COLLECTION] Malicious clients: {json_data.get('malicious_count', 0)}, Benign clients: {json_data.get('benign_count', 0)}")
        print(f"[GRADIENT COLLECTION] Trimmed each gradient to first {MAX_GRADIENT_SIZE} elements for visualization")
    # ========== END NEW CODE ==========
    
    model.set_weights(weights)


