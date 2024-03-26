import numpy as np
import random 
import pandas as pd
import sys
import os 
import argparse 
from utils import regress_score, regress_score_select, compute_T_df, compute_T_other_df, compute_T_other_2_df, compute_left_interval, compute_right_interval 

parser = argparse.ArgumentParser('')

parser.add_argument('--seed', type=int, default=1, help='random seed for running experiment')
parser.add_argument('--alpha', type=int, default=1, choices = [1,2,3,4,5,6,7,8,9], help='nominal coverage level') 
parser.add_argument('--output_dir', type=str, required=True, help='output directory for results')  

args = parser.parse_args()
seed = args.seed
SAVE_PATH = args.output_dir
q_ind = args.alpha 
score_pred = 'residual' 

if not os.path.exists(SAVE_PATH):
  os.makedirs(SAVE_PATH)
   

# =============================================================================
# ## read in data
# ============================================================================= 
 
if score_pred == 'residual':
    data = np.load("../data/icu_var.npy")
    data = pd.DataFrame(data)
    data.columns = ['y', 'pred', 'std']
    data['std'] = np.sqrt(data['std'])
    data['pred'] = data['pred'] + np.random.uniform(size=data.shape[0]) * 0.01
    data['y'] = data['y'] + np.random.uniform(size=data.shape[0]) * 0.01

data = data.reset_index(drop = True)

# data split
     
np.random.seed(seed)
n = data.shape[0]
reind = np.random.permutation(n)

data_calib = data.iloc[reind[0:3000]].copy(deep=True)
data_calib = data_calib.reset_index(drop=True)
data_test = data.iloc[reind[3000:5000]].copy(deep=True)
data_test = data_test.reset_index(drop=True)
    
    
#=============================================================================
# # preliminary conformal prediction set  (|y-mu|/sigma)
#=============================================================================

calib_score_pred = np.abs(data_calib['y'] - data_calib['pred']) / data_calib['std']
hat_eta = np.quantile(np.array(list(calib_score_pred)+[np.Inf]), 
                      1-q_ind/10, 
                      interpolation='higher') 

# test score
test_score_pred = np.abs(data_test['y'] - data_test['pred']) / data_test['std']
# prediction interval 
itv_upp = np.array(data_test['pred']) + hat_eta * np.array(data_test['std'])
itv_low = np.maximum(0, np.array(data_test['pred']) - hat_eta * np.array(data_test['std']))
itv_len = itv_upp - itv_low
        
# =============================================================================
#   find selected units
# =============================================================================
UPPER_BOUND = 6
selected_test_idx = np.arange(0, 2000)[itv_upp <= UPPER_BOUND] 

if len(selected_test_idx) > 0:
    selected_test_score = np.array(test_score_pred[selected_test_idx])
        
    # =============================================================================
    #   find correct prediction set
    # =============================================================================
    
    # compute {y: eta- <= S(X_n+1, y) <= eta+ }
    K = int(np.ceil((1-q_ind/10) * 3001))
    eta_pos = np.sort(calib_score_pred)[K]
    eta_neg = np.sort(calib_score_pred)[K-2]
    range_set1 = [eta_neg, eta_pos]
        
    # compute {y: V <= q1, S <= eta-}   
            
    calib_upp = data_calib['pred'] + hat_eta * data_calib['std']
    calib_low = np.maximum(0, 
                           data_calib['pred'] - hat_eta * data_calib['std'])
    calib_len = calib_upp - calib_low
    
    calib_upp_pos = data_calib['pred'] + eta_pos * data_calib['std']
    calib_low_pos = np.maximum(0, 
                           data_calib['pred'] - eta_pos * data_calib['std'])
    calib_len_pos = calib_upp_pos - calib_low_pos
    
    calib_upp_neg = data_calib['pred'] + eta_neg * data_calib['std']
    calib_low_neg = np.maximum(0, 
                           data_calib['pred'] - eta_neg * data_calib['std'])
    calib_len_neg = calib_upp_neg - calib_low_neg
    
    if_in_q1 = 1 * (calib_score_pred <= eta_neg) * (calib_upp <= UPPER_BOUND) + 1 * (calib_score_pred > eta_neg) * (calib_upp_neg <= UPPER_BOUND)
    if_in_q2 = 1 * (calib_score_pred <= hat_eta) * (calib_upp_pos <= UPPER_BOUND) + 1 * (calib_score_pred > hat_eta) * (calib_upp <= UPPER_BOUND)
    
    q1 = np.quantile(np.array(list(calib_score_pred[if_in_q1==1])+ [np.Inf]),
                     1-q_ind/10, 
                     interpolation = 'higher')
    q2 = np.quantile(np.array(list(calib_score_pred[if_in_q2==1])+ [np.Inf]),
                     1-q_ind/10, 
                     interpolation = 'higher') 
    
    # combine subsets in the prediction set

    range_set2 = [-np.Inf, np.minimum(q1, eta_neg)]
    range_set3 = [eta_pos, q2]
     
    cover_jomi = 1 * (selected_test_score >= range_set1[0]) * (selected_test_score <= range_set1[1]) + 1 * (selected_test_score >= range_set2[0]) * (selected_test_score <= range_set2[1]) + 1 * (selected_test_score >= range_set3[0]) * (selected_test_score <= range_set3[1])
    
    # merge ranges to produce length 

    if q2 > eta_pos: 
        if q1 >= eta_neg:
            size_jomi = np.array(data_test['pred'] + q2 * data_test['std'] - np.maximum(0, data_test['pred'] - q2 * data_test['std']))
        else: 
            size_jomi = np.array(data_test['pred'] + q1 * data_test['std'] - np.maximum(0, data_test['pred'] - q1 * data_test['std']) + 2 * (q2 - eta_neg) * data_test['std'])
    else:
        if q1 >= eta_neg:
            size_jomi = np.array(data_test['pred'] + eta_pos * data_test['std'] - np.maximum(0, data_test['pred'] - eta_pos * data_test['std']))
        else:
            size_jomi = np.array(data_test['pred'] + q1 * data_test['std'] - np.maximum(0, data_test['pred'] - q1 * data_test['std']) + 2 * (eta_pos - eta_neg) * data_test['std'])
         
    size_jomi = size_jomi[selected_test_idx]
    miscover_jomi = 1 * (cover_jomi == 0)
    miscover_naive = 1 * (selected_test_score > hat_eta)
    size_naive = itv_len[selected_test_idx]
    
    results = pd.concat([pd.DataFrame({"miscover": miscover_jomi, 
                                       "size": size_jomi, 
                                       "num_select": len(selected_test_idx),
                                       "method": "JOMI",
                                       "marginal": 1-q_ind/10,
                                       "seed": seed, 
                                       "q1": q1,
                                       "q2": q2,
                                       "hat_eta": hat_eta,
                                       "eta_pos": eta_pos,
                                       "eta_neg": eta_neg}),
                         pd.DataFrame({"miscover": miscover_naive, 
                                       "size": size_naive, 
                                       "num_select": len(selected_test_idx),
                                       "method": "Vanilla_CP",
                                       "marginal": 1-q_ind/10,
                                       "seed": seed, 
                                       "q1": q1,
                                       "q2": q2,
                                       "hat_eta": hat_eta,
                                       "eta_pos": eta_pos,
                                       "eta_neg": eta_neg})], axis=0)
            
     
        
    
results.to_csv(SAVE_PATH+"/lower_seed_"+str(seed)+"_alpha_"+str(q_ind)+".csv")   
      
