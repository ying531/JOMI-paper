import numpy as np
import random 
import pandas as pd
import sys
import os 
import argparse
from utils import regress_score, regress_score_select, compute_T_df, compute_T_other_df, compute_T_other_2_df, compute_left_interval, compute_right_interval

parser = argparse.ArgumentParser('')


parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--alpha', type=int, default=1, 
    choices = [1,2,3,4,5,6,7,8,9], help='nominal miscoverage level (times 10)') 
parser.add_argument('--output_dir', type=str, 
    required=True, help='output directory for saving results') 

args = parser.parse_args()
seed = args.seed
SAVE_PATH = args.output_dir
q_ind = args.alpha 
score_pred = 'residual'
c_threshold = 'quantile'
 
# SAVE_PATH='/workspace/results/'

if not os.path.exists(SAVE_PATH):
  os.makedirs(SAVE_PATH)
  

 
raw_calib = pd.read_csv("../results/DTI/CNN_transformer_calib_"+str(seed % 10 + 1)+".csv") 
raw_test = pd.read_csv("../results/DTI/CNN_transformer_test_"+str(seed % 10 + 1)+".csv")
raw_test.columns = raw_calib.columns
data = pd.concat([raw_calib, raw_test], axis=0).iloc[:,1:raw_calib.shape[1]]
data = data.rename({"calib_pred": "pred", "calib_true": "y"}, axis = "columns")


results = pd.DataFrame() 
 
# =============================================================================
#   set random seed and data splitting
# =============================================================================
np.random.seed(seed)
n = data.shape[0]
reind = np.random.permutation(n)

data_calib = data.iloc[reind[0:int(n/2)]].reset_index().iloc[:,1:data.shape[1]]
data_test = data.iloc[reind[int(n/2):n]].reset_index().iloc[:,1:data.shape[1]]

# =============================================================================
#   conformal prediction set
# =============================================================================
calib_y = np.array(data_calib['y'])
calib_mu = np.array(data_calib['pred'])
calib_score_pred = regress_score(calib_y, calib_mu, score = score_pred)
n_calib = len(calib_y)
hat_eta = np.quantile(np.array(list(calib_score_pred)+[np.Inf]), 
                      1-q_ind/10, 
                      interpolation='higher')
# test scores
test_mu = np.array(data_test['pred']) 
test_y = np.array(data_test['y'])
test_score_pred = regress_score(test_y, test_mu, score = score_pred)
n_test = len(test_mu)

# thresholds c_i
c_calib = np.array(data_calib['q7'])
c_test = np.array(data_test['q7']) 
 
# score for selection
calib_score_for_selection = regress_score_select(calib_y, calib_mu, 'clip', if_test=True, c=c_calib) 
test_score_for_selection = regress_score_select(test_y, test_mu, 'clip', if_test=True, c=c_test)  

# =============================================================================
#   JOMI for conformalized selection
# =============================================================================

# dataframe as preparation    
all_df = compute_T_df(calib_score_low = calib_score_for_selection[calib_y <= c_calib], 
                      test_score = test_score_for_selection, 
                      n_calib = n_calib, n_test = n_test) 
 
sel_qs = np.linspace(0.1, 0.9, num = 9)

results = pd.DataFrame()
for qsel in sel_qs:

    # conformal selection to determine selection set
    idx_smaller = [i for i in range(all_df.shape[0]) if all_df['F'].iloc[i] <= qsel]
    
    if len(idx_smaller) > 0:
        empty_select = 0

        # stopping time T for selection
        # selection rule is S <= T
        TT = all_df['S'].iloc[int(np.max(idx_smaller))]
        test_sel_idx = [i for i in range(n_test) if test_score_for_selection[i] <= TT]
        n_select = len(test_sel_idx)
        
        pval = [] 
        pval_rand = [] 
        num_itv = []
        num_itv_rand = []
        len_itv = []
        len_itv_rand = []
        n_ref = []
        for j in test_sel_idx:
            df_T1 = compute_T_other_df(calib_score_low = calib_score_for_selection[calib_y <= c_calib], 
                                    test_score_loo = np.array(list(test_score_for_selection)[:j] + list(test_score_for_selection)[(j+1):]), 
                                    n_calib = n_calib, n_test = n_test)
            idx_smaller_T1 = [i for i in range(df_T1.shape[0]) if df_T1['F'].iloc[i] <= qsel] 
            T1 = df_T1['S'].iloc[int(np.max(idx_smaller_T1))] if len(idx_smaller_T1) > 0 else np.Inf 
            
            df_T2 = compute_T_other_2_df(calib_score_low = calib_score_for_selection[calib_y <= c_calib], 
                                    test_score_loo = np.array(list(test_score_for_selection)[:j] + list(test_score_for_selection)[(j+1):]), 
                                    n_calib = n_calib, n_test = n_test)
            idx_smaller_T2 = [i for i in range(df_T2.shape[0]) if df_T2['F'].iloc[i] <= qsel] 
            T2 = df_T2['S'].iloc[int(np.max(idx_smaller_T2))] if len(idx_smaller_T2) > 0 else np.Inf 
            
            # R_j+ = {k: S_k >= T1} 
            R_pos = [k for k in range(n_calib) if calib_score_for_selection[k] <= T1 and calib_y[k] > c_calib[k]] + [k for k in range(n_calib) if calib_score_for_selection[k] <= T2 and calib_y[k] <= c_calib[k]]
            R_neg = [k for k in range(n_calib) if calib_score_for_selection[k] <= T1 and calib_y[k] > c_calib[k]] + [k for k in range(n_calib) if calib_score_for_selection[k] <= TT and calib_y[k] <= c_calib[k]]
            
            R_j = R_pos if test_y[j] > c_test[j] else R_neg
            n_ref = n_ref + [len(R_j)]
            
            # =======
            # realized p_j(Y_{n+j})
            # ======= 
            if len(R_j) > 0: # calibration reference set is non-empty
                ref_calib_score = calib_score_pred[R_j] 
                pval = pval + [(1+np.sum(test_score_pred[j] <= ref_calib_score)) / (1+len(R_j))]
                pval_rand = pval_rand + [(np.sum(test_score_pred[j] < ref_calib_score) + (1+np.sum(ref_calib_score == test_score_pred[j])) * np.random.uniform(size=1)[0]) / (1+len(R_j))]
            else:
                pval = pval + [0]
                pval_rand = pval_rand + [0] 
                
            # =======
            # invert p_j(y) for prediction intervals
            # =======  
            hat_eta_sel_pos = np.quantile(list(calib_score_pred[R_pos]) + [np.Inf],
                                          1-q_ind/10, 
                                          interpolation='higher') if len(R_pos)>0 else np.Inf 
            hat_eta_sel_neg = np.quantile(list(calib_score_pred[R_neg]) + [np.Inf],
                                          1-q_ind/10, 
                                          interpolation='higher') if len(R_neg)>0 else np.Inf 
            itv_left = compute_left_interval(test_mu[j], hat_eta_sel_neg, c_test[j])
            itv_right = compute_right_interval(test_mu[j], hat_eta_sel_pos, c_test[j])
            
            if itv_left is None or itv_right is None:
                itv = itv_right if itv_left is None else itv_left  
                num_itv = num_itv + [1]
                len_itv = len_itv + [itv[1] - itv[0]]
            elif itv_left[1] < itv_right[0]:
                itv = list(itv_left) + list(itv_right)
                num_itv = num_itv + [2]
                len_itv = len_itv + [itv_left[1] - itv_left[0] + itv_right[1] - itv_right[0]]
            else:
                itv = np.array([itv_left[0], itv_right[1]])
                num_itv = num_itv + [1]
                len_itv = len_itv + [itv[1] - itv[0]]
            
            
            # =======
            # invert p_j(y)-rand for prediction intervals
            # ======= 
            Uj = np.random.uniform(size=1)[0]
            if len(R_pos) > 0:
                V_Rj_pos = np.sort(calib_score_pred[R_pos])
                pj_rand_pos_all = [(np.sum(V_Rj_pos > V_Rj_pos[ii]) + (1+np.sum(V_Rj_pos==V_Rj_pos[ii]))*Uj) / (1+len(R_pos)) for ii in range(len(R_pos))]
                V_Rj_pos_idx = np.array(pj_rand_pos_all) >= q_ind/10
                if np.sum(V_Rj_pos_idx) > 0:
                    hat_eta_sel_pos_rand = np.max(V_Rj_pos[V_Rj_pos_idx])
                else:
                    hat_eta_sel_pos_rand = np.Inf
                    print(pj_rand_pos_all) 
            else:
                hat_eta_sel_pos_rand = np.Inf 
                
            if len(R_neg) > 0:
                V_Rj_neg = np.sort(calib_score_pred[R_neg])
                pj_rand_neg_all = [(np.sum(V_Rj_neg > V_Rj_neg[ii]) + (1+np.sum(V_Rj_neg==V_Rj_neg[ii]))*Uj) / (1+len(R_neg)) for ii in range(len(R_neg))] 
                V_Rj_neg_idx = np.array(pj_rand_neg_all) >= q_ind/10
                if np.sum(V_Rj_neg_idx) > 0:
                    hat_eta_sel_neg_rand = np.max(V_Rj_neg[V_Rj_neg_idx])
                else:
                    hat_eta_sel_neg_rand = np.Inf
                    print(pj_rand_neg_all) 
            else: 
                hat_eta_sel_neg_rand = np.Inf
             

            # =======
            # merge segments for final prediction set
            # =======
            itv_left_rand = compute_left_interval(test_mu[j], hat_eta_sel_neg_rand, c_test[j])
            itv_right_rand = compute_right_interval(test_mu[j], hat_eta_sel_pos_rand, c_test[j])
            
            if itv_left_rand is None or itv_right_rand is None:
                itv_rand = itv_right_rand if itv_left_rand is None else itv_left_rand  
                num_itv_rand = num_itv_rand + [1]
                len_itv_rand = len_itv_rand + [itv_rand[1] - itv_rand[0]]
            elif itv_left_rand[1] < itv_right_rand[0]:
                itv = list(itv_left_rand) + list(itv_right_rand)
                num_itv_rand = num_itv_rand + [2]
                len_itv_rand = len_itv_rand + [itv_left_rand[1] - itv_left_rand[0] + itv_right_rand[1] - itv_right_rand[0]]
            else:
                itv_rand = np.array([itv_left_rand[0], itv_right_rand[1]])
                num_itv_rand = num_itv_rand + [1]
                len_itv_rand = len_itv_rand + [itv_rand[1] - itv_rand[0]]
        
        
        # =======
        # evaluate miscoverage, size of prediction set, and num of segments
        # =======

        sel_miscover = np.mean(np.array(pval) < q_ind/10)
        sel_miscover_rand = np.mean(np.array(pval_rand) < q_ind/10) 
        sel_size = np.mean(len_itv)
        sel_itv_num = np.mean(num_itv)
        sel_size_rand = np.mean(len_itv_rand)
        sel_itv_num_rand = np.mean(num_itv_rand)
        naive_miscover = np.mean(test_score_pred[test_sel_idx] > hat_eta) 
        naive_size = 2 * hat_eta
        num_ref = np.mean(np.array(n_ref))
         
        
    else: # no selection
        n_select = 0
        empty_select = 1 
        num_ref = None
        sel_miscover = None
        sel_miscover_rand = None
        sel_size = None
        sel_size_rand = None 
        sel_itv_num = None
        sel_itv_num_rand = None
        naive_miscover = None
        naive_size = None  
        
        
    results = pd.concat([results, pd.DataFrame({"miscover": [sel_miscover, 
                                                             sel_miscover_rand, naive_miscover], 
                                               "size": [sel_size, sel_size_rand, naive_size], 
                                               "num_itv": [sel_itv_num, sel_itv_num_rand, 1],
                                               "num_select": n_select,
                                               "num_ref": num_ref,
                                               "method": ['JOMI', 'JOMI_rand', 'naive'], 
                                               "marginal": 1- q_ind/10,  
                                               "select_FDR": qsel,
                                               'seed': seed,
                                               "empty_select": empty_select})], axis=0)
         
      
results.to_csv(SAVE_PATH+"/confsel_DTI_seed_"+str(seed)+"_alpha_"+str(q_ind)+".csv") 