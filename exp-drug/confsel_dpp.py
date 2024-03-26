import numpy as np
import random 
import pandas as pd
import sys
import os 
import argparse
from utils import binary_score_select, binary_score, compute_T_df, compute_T_other_df, compute_T_other_2_df

parser = argparse.ArgumentParser('')

parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--alpha', type=int, default=1, 
    choices = [1,2,3,4,5,6,7,8,9], help='nominal miscoverage level (times 10)') 
parser.add_argument('--output_dir', type=str, 
    required=True, help='output directory for saving results')  
parser.add_argument('--pred', type=str, default='bin', 
    choices = ['bin', 'APS'], 
    help='nonconformity score for prediction interval, bin for binary score, APS for the APS score')

args = parser.parse_args()
seed = args.seed
SAVE_PATH = args.output_dir
q_ind = args.alpha 
score_pred = args.pred
  

if not os.path.exists(SAVE_PATH):
  os.makedirs(SAVE_PATH)
   
data = pd.read_csv("../results/DPP/DPP_"+str(seed % 10 + 1)+".csv")
data = pd.DataFrame({"pred": list(data['calib_pred']) + list(data['test_pred']),
                     "y": list(data['calib_true']) + list(data['test_true'])})

# =============================================================================
# random seed and data split# =============================================================================

np.random.seed(seed)
n = data.shape[0]
reind = np.random.permutation(n)

data_calib = data.iloc[reind[0:int(n/2)]].reset_index().iloc[:,1:3]
data_test = data.iloc[reind[int(n/2):n]].reset_index().iloc[:,1:3]

# conformal prediction set 
calib_y = np.array(data_calib['y'])
calib_mu = np.array(data_calib['pred'])
calib_score = binary_score(calib_y, calib_mu, score_pred)  
n_calib = len(calib_y)
# quantile for vanilla conformal prediction
hat_eta = np.quantile(np.array(list(calib_score)+[np.Inf]), 
                      1-q_ind/10, 
                      interpolation='higher')
 
# test scores
test_mu = np.array(data_test['pred'])
n_test = len(test_mu)
test_y = np.array(data_test['y'])
test_score = binary_score(test_y, test_mu, score_pred)  
test_score_1 = binary_score(1, test_mu, score_pred)  
test_score_0 = binary_score(0, test_mu, score_pred)  

set_1_in = 1 * (test_score_1 <= hat_eta)
set_0_in = 1 * (test_score_0 <= hat_eta) 
set_test = set_1_in + set_1_in * set_0_in - (1-set_1_in) * (1-set_0_in)

# score for selection
calib_score_for_selection = binary_score_select(0, calib_mu, 'clip', if_test=True, c=0.5) 
test_score_for_selection = binary_score_select(0, test_mu, 'clip', if_test=True, c=0.5)
    
# =============================================================================
#   JOMI for conformalized selection
# =============================================================================
     
all_df = compute_T_df(calib_score_low = calib_score_for_selection[calib_y <= 0.5], 
                      test_score = test_score_for_selection, 
                      n_calib = n_calib, n_test = n_test) 

sel_qs = np.linspace(0.05, 0.3, num = 6)
results = pd.DataFrame()
for qsel in sel_qs:

    # compute selection set
    idx_smaller = [i for i in range(all_df.shape[0]) if all_df['F'].iloc[i] <= qsel]
    
    if len(idx_smaller) > 0:

        empty_select = 0 
        TT = all_df['S'].iloc[int(np.max(idx_smaller))]
        test_sel_idx = [i for i in range(n_test) if test_score_for_selection[i] <= TT] 
        n_select = len(test_sel_idx)
        
        pval = [] 
        pval_rand = []
        sel_test_set = []
        sel_test_set_rand = []
        for j in test_sel_idx:

            df_T1 = compute_T_other_df(calib_score_low = calib_score_for_selection[calib_y <= 0.5], 
                                    test_score_loo = np.array(list(test_score_for_selection)[:j] + list(test_score_for_selection)[(j+1):]), 
                                    n_calib = n_calib, n_test = n_test)
            idx_smaller_T1 = [i for i in range(df_T1.shape[0]) if df_T1['F'].iloc[i] <= qsel] 
            T1 = df_T1['S'].iloc[int(np.max(idx_smaller_T1))] if len(idx_smaller_T1) > 0 else np.Inf 
            
            df_T2 = compute_T_other_2_df(calib_score_low = calib_score_for_selection[calib_y <= 0.5], 
                                    test_score_loo = np.array(list(test_score_for_selection)[:j] + list(test_score_for_selection)[(j+1):]), 
                                    n_calib = n_calib, n_test = n_test)
            idx_smaller_T2 = [i for i in range(df_T2.shape[0]) if df_T2['F'].iloc[i] <= qsel] 
            T2 = df_T2['S'].iloc[int(np.max(idx_smaller_T2))] if len(idx_smaller_T2) > 0 else np.Inf 
            
            # R_j+ = {k: S_k >= T1} 
            R_pos = [k for k in range(n_calib) if calib_score_for_selection[k] <= T1 and calib_y[k] > 0.5] + [k for k in range(n_calib) if calib_score_for_selection[k] <= T2 and calib_y[k] <= 0.5]
            R_neg = [k for k in range(n_calib) if calib_score_for_selection[k] <= T1 and calib_y[k] > 0.5] + [k for k in range(n_calib) if calib_score_for_selection[k] <= TT and calib_y[k] <= 0.5]
            
            R_j = R_pos if test_y[j] > 0.5 else R_neg
            
            # calibration reference set is non-empty
            if len(R_j) > 0: 
                ref_calib_score = calib_score[R_j] 
                
                # realized p_j(Y_{n+j})
                pval = pval + [(1+np.sum(test_score[j] <= ref_calib_score)) / (1+len(R_j))]
                pval_rand = pval_rand + [(np.sum(test_score[j] < ref_calib_score) + (1+np.sum(ref_calib_score == test_score[j])) * np.random.uniform(size=1)[0]) / (1+len(R_j))]
                
                # hypothesize p_j(0) and p_j(1)
                Uj = np.random.uniform(size=1)[0]
                pj_0 = (1+np.sum(ref_calib_score <= test_score_0[j])) / (1+len(R_j))
                pj_1 = (1+np.sum(ref_calib_score <= test_score_1[j])) / (1+len(R_j))
                pj_rand_0 = (np.sum(ref_calib_score < test_score_0[j]) + (1+np.sum(ref_calib_score == test_score_0[j])) * Uj) / (1+len(R_j))
                pj_rand_1 = (np.sum(ref_calib_score < test_score_1[j]) + (1+np.sum(ref_calib_score == test_score_1[j])) * Uj) / (1+len(R_j))
                
                if pj_0 <= 1 - q_ind/10:
                    sel_test_set = sel_test_set + [2] if pj_1 <= 1 - q_ind/10 else sel_test_set + [0]
                else:
                    sel_test_set = sel_test_set + [1] if pj_1 <= 1 - q_ind/10 else sel_test_set + [-1]
                
                if pj_rand_0 <= 1 - q_ind/10:
                    sel_test_set_rand = sel_test_set_rand + [2] if pj_rand_1 <= 1 - q_ind/10 else sel_test_set_rand + [0]
                else:
                    sel_test_set_rand = sel_test_set_rand + [1] if pj_rand_1 <= 1 - q_ind/10 else sel_test_set_rand + [-1]
            else:
                pval = pval + [1]
                pval_rand = pval_rand + [1]
                sel_test_set = sel_test_set + [2]
                sel_test_set_rand = sel_test_set_rand + [2] 
            
        # evaluate miscoverage and size of prediction set
        sel_miscover = np.mean(np.array(pval) < q_ind/10) 
        sel_miscover_rand = np.mean(np.array(pval_rand) < q_ind/10) 
        sel_size = 1 + np.mean(np.array(sel_test_set) == 2) - np.mean(np.array(sel_test_set)==-1)
        sel_size_rand = 1 + np.mean(np.array(sel_test_set_rand) == 2) - np.mean(np.array(sel_test_set_rand)==-1)
        naive_miscover = np.mean((set_test[test_sel_idx]<2)*(set_test[test_sel_idx] != test_y[test_sel_idx]))
        naive_size = 1 + np.mean(set_test[test_sel_idx] == 2) - np.mean(set_test[test_sel_idx]==-1) 
    
    else:
        sel_miscover = None
        sel_size = None
        sel_miscover_rand = None
        sel_size_rand = None
        naive_miscover = None
        naive_size = None
        select_eta = None 
        n_select = 0  
     

    results = pd.concat([results, pd.DataFrame({"miscover": [sel_miscover, 
                                                             sel_miscover_rand, naive_miscover], 
                                               "size": [sel_size, sel_size_rand, naive_size], 
                                               "method": ['JOMI', 'JOMI_rand', 'naive'],
                                              "marginal": 1- q_ind/10,
                                              "select_FDR": qsel,
                                              'seed': seed,
                                              "num_select": n_select, 
                                              "score_pred": score_pred})], axis=0)
         
 
results.to_csv(SAVE_PATH+"/confsel_DPP_"+score_pred+"_seed_"+str(seed)+"_alpha_"+str(q_ind)+".csv") 

