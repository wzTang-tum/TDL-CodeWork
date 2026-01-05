import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

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
    norm = np.linalg.norm(weights)
    if norm > delta and sigma>0:
        weights = weights*delta/norm
    if sigma>0:
        weights = weights + np.random.normal(0,sigma,size=weights.shape)
    return weights

def renorm(w1,w2):
    n1 = np.linalg.norm(w1)
    n2 = np.linalg.norm(w2)
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

def RobustAggergation(mode,all_weights,old_weights, f=0):
    flaten_all_weights = all_weights
    selected_indices = np.arange(len(all_weights)) # Default: all selected
    n = len(all_weights)
    
    if mode == 'FedAvg':
        agg_weights = np.mean(flaten_all_weights,axis=0)
    
    elif mode == 'Mid':
        agg_weights = np.median(flaten_all_weights, axis=0)
        
    elif mode == 'Krum':
        # Ensure f is within valid range for Krum: n >= 2f + 3 => 2f <= n - 3 => f <= (n-3)/2
        # If f is too large, adjust it
        if n < 2 * f + 3:
            effective_f = max(0, (n - 3) // 2)
        else:
            effective_f = f
            
        if effective_f < 0: effective_f = 0
        
        # Number of neighbors to consider: n - f - 2
        k = n - effective_f - 2
        if k <= 0: k = 1 # Fallback
        
        # Compute pairwise squared Euclidean distances
        sq_dists = cdist(flaten_all_weights, flaten_all_weights, 'sqeuclidean')
        
        scores = []
        for i in range(n):
            # Sort distances
            dists = np.sort(sq_dists[i])
            # Sum of closest k distances (excluding self which is 0)
            # dists[0] is self (0)
            # We take dists[1 : k+1]
            score = np.sum(dists[1 : k+1])
            scores.append(score)
            
        best_idx = np.argmin(scores)
        agg_weights = flaten_all_weights[best_idx]
        selected_indices = [best_idx]
        
    elif mode == 'MKrum':
        # Multi-Krum: Select m candidates using Krum score, then average
        # Usually m = n - f
        m = n - f
        if m <= 0: m = 1
        
        if n < 2 * f + 3:
            effective_f = max(0, (n - 3) // 2)
        else:
            effective_f = f
            
        k = n - effective_f - 2
        if k <= 0: k = 1
        
        sq_dists = cdist(flaten_all_weights, flaten_all_weights, 'sqeuclidean')
        scores = []
        for i in range(n):
            dists = np.sort(sq_dists[i])
            score = np.sum(dists[1 : k+1])
            scores.append(score)
            
        # Select top m indices with smallest scores
        sorted_indices = np.argsort(scores)
        best_indices = sorted_indices[:m]
        
        agg_weights = np.mean(flaten_all_weights[best_indices], axis=0)
        selected_indices = best_indices
        
    elif mode == 'Norm':
        # Norm Filtering: Remove f updates with largest L2 norms
        norms = np.linalg.norm(flaten_all_weights, axis=1)
        # Sort indices by norm
        sorted_indices = np.argsort(norms)
        # Keep n - f smallest
        num_keep = n - f
        if num_keep <= 0: num_keep = 1
        
        selected_indices = sorted_indices[:num_keep]
        agg_weights = np.mean(flaten_all_weights[selected_indices], axis=0)
        
    elif mode == 'Contra':
        # Assuming Contra refers to Trimmed Mean (Coordinate-wise)
        # Trim beta fraction from each end. beta = f / n
        beta = f / n
        if beta >= 0.5: beta = 0.49 # Limit trimming
        
        # Sort along axis 0 (clients)
        sorted_weights = np.sort(flaten_all_weights, axis=0)
        
        # Number to trim from each end
        k = int(beta * n)
        
        # Slice
        if k > 0:
            trimmed_weights = sorted_weights[k:-k]
        else:
            trimmed_weights = sorted_weights
            
        agg_weights = np.mean(trimmed_weights, axis=0)
        
    elif mode =='RobustDPFL':
        z = np.abs(flaten_all_weights.mean(axis=-1))
        
        y_pred = KMeans(n_clusters=2, random_state=9).fit_predict(z.reshape((-1,1)))
        index0 = np.where(y_pred==0)[0]
        index1 = np.where(y_pred==1)[0]
        if z[index0].mean()<z[index1].mean():
            index = index0
        else:
            index = index1
        agg_weights = flaten_all_weights[index].mean(axis=0)
        selected_indices = index
                
    agg_weights = func_unflatten_weights(agg_weights,old_weights)
    
    return agg_weights, selected_indices


def FL(attack_mode,mode,user_num,model,taxic_clients,train_users,train_images,train_labels,sigma=0.1,delta=10,intre_epoch=1, round_idx=None, total_epochs=None):

    user_indexs = np.random.permutation(len(train_users))[:user_num]
    
    old_weights = model.get_weights()
    flatten_old_weights = func_flatten_weights(old_weights)
    all_weights = []
    
    # ========== NEW CODE: Initialize gradient collection for last round ==========
    # Record at round 10 (index 9)
    is_last_round = (round_idx == 0)
    collected_gradients = {
        "malicious": [],
        "benign": [],
        "selected": None
    }
    client_types = [] # Track client types for mask generation
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
            elif attack_mode == 'AttackFL':
                # AttackFL: Conventional FL without differential privacy
                # No gradient clipping or DP noise - direct upload of raw gradients
                # This creates the baseline "Attack-FL" from the paper (Figure 1)
                weights = delta_weights+flatten_old_weights
                all_weights.append(weights)
        else:
            # Benign clients
            if attack_mode != 'AttackFL':
                # Apply DP noise for all modes except AttackFL
                delta_weights = LDP(delta_weights,sigma,delta)
            weights = delta_weights+flatten_old_weights
            all_weights.append(weights)
            
        # ========== NEW CODE: Collect gradient if last round ==========
        if is_last_round:
            final_delta = weights - flatten_old_weights
            if ui in taxic_clients:
                collected_gradients["malicious"].append(final_delta.copy())
                client_types.append('malicious')
            else:
                collected_gradients["benign"].append(final_delta.copy())
                client_types.append('benign')
        # ========== END NEW CODE ==========

        model.set_weights(old_weights)

    all_weights = np.array(all_weights)
    
    # Estimate f (number of attackers) for robust aggregation methods
    total_n = len(train_users)
    total_m = len(taxic_clients)
    ratio = total_m / total_n if total_n > 0 else 0
    f = int(np.ceil(len(all_weights) * ratio))
    
    weights, selected_indices = RobustAggergation(mode,all_weights,old_weights, f)    
    
    # ========== NEW CODE: Collect selected gradients from aggregation ==========
    if is_last_round:
        # Extract indices of selected gradients from aggregation
        # (RobustDPFL uses KMeans clustering; we extract the selected cluster)
        # Note: For simplicity, we're storing the aggregated gradient as "selected"
        
        # Convert to lists for JSON serialization
        
        json_data = {}
        
        # Process malicious gradients
        if collected_gradients["malicious"]:
            malicious_processed = []
            for g in collected_gradients["malicious"]:
                grad_arr = np.array(g)
                # round to 3 decimals
                grad_arr = np.round(grad_arr, 3)
                malicious_processed.append(grad_arr.tolist())
            json_data["malicious"] = malicious_processed
            json_data["malicious_count"] = len(collected_gradients["malicious"])

        # Process benign gradients
        if collected_gradients["benign"]:
            benign_processed = []
            for g in collected_gradients["benign"]:
                grad_arr = np.array(g)
                grad_arr = np.round(grad_arr, 3)  # round to 3 decimals
                benign_processed.append(grad_arr.tolist())
            json_data["benign"] = benign_processed
            json_data["benign_count"] = len(collected_gradients["benign"])

        # Process selected gradient
        selected_list = [g.tolist() if hasattr(g, 'tolist') else g for g in weights]
        selected_processed = []
        for g in selected_list:
            # g may be a list or scalar
            if isinstance(g, list):
                grad_arr = np.array(g)
                grad_arr = np.round(grad_arr, 3)  # round to 3 decimals
                selected_processed.append(grad_arr.tolist())
            else:
                # leave non-list items as-is
                selected_processed.append(g)
        json_data["selected"] = selected_processed
        
        # --- NEW: Save selection masks based on actual aggregation decision ---
        selected_indices_set = set(selected_indices)
        malicious_mask = []
        benign_mask = []
        
        # client_types matches all_weights indices
        for i, c_type in enumerate(client_types):
            is_selected = 1 if i in selected_indices_set else 0
            if c_type == 'malicious':
                malicious_mask.append(is_selected)
            else:
                benign_mask.append(is_selected)
                
        json_data["malicious_selected_mask"] = malicious_mask
        json_data["benign_selected_mask"] = benign_mask
        # ------------------------------------------------------------------
        
        # Create output directory if not exists
        output_dir = "../Result/gradient_analysis"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSON format
        output_json = os.path.join(output_dir, "gradients_last_round.json")
        
        with open(output_json, 'w') as f:
            json.dump(json_data, f)
        
        print(f"[GRADIENT COLLECTION] Saved gradients to {output_json}")
        print(f"[GRADIENT COLLECTION] Malicious clients: {json_data.get('malicious_count', 0)}, Benign clients: {json_data.get('benign_count', 0)}")
        print(f"[GRADIENT COLLECTION] Saved full gradients (rounded to 3 decimals)")
    # ========== END NEW CODE ==========
    
    model.set_weights(weights)


