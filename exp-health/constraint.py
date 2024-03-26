import numpy as np
import random 
import pandas as pd
import sys
import os 
import argparse 
from utils import regress_score, regress_score_select, compute_T_df, compute_T_other_df, compute_T_other_2_df, compute_left_interval, compute_right_interval
from mknapsack import solve_single_knapsack 

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

data = pd.read_csv("../data/icu_pred.csv")
data.columns = ['y', 'pred']

# generate cost
np.random.seed(1)
data = data.iloc[np.random.choice(data.shape[0], 
                                  5000, replace=False)]
data['cost'] = np.ceil(np.exp(data['pred']/np.max(data['pred'])) + np.abs(np.sin(data['pred'])) + np.random.exponential(size = data.shape[0]) - 1 + np.random.uniform(size = data.shape[0]))

# split data
data = data.reset_index(drop = True)
  
np.random.seed(seed)
n = data.shape[0]
reind = np.random.permutation(n)

data_calib = data.iloc[reind[0:3000]].copy(deep=True)
data_calib = data_calib.reset_index(drop=True)
data_test = data.iloc[reind[3000:5000]].copy(deep=True)
data_test = data_test.reset_index(drop=True)

# conformal prediction set 
calib_y = np.array(data_calib['y'])
calib_mu = np.array(data_calib['pred'])
calib_score_pred = regress_score(calib_y, calib_mu, score = score_pred)
n_calib = len(calib_y)
hat_eta = np.quantile(np.array(list(calib_score_pred)+[np.Inf]), 
                      1-q_ind/10, 
                      interpolation='higher')

test_mu = np.array(data_test['pred']) 
test_y = np.array(data_test['y'])
test_score_pred = regress_score(test_y, test_mu, score = score_pred)
n_test = len(test_mu)

all_score_pred = regress_score(data['y'], data['pred'], score = score_pred)

# interested threshold c_i  
excess_calib = calib_mu
excess_test = test_mu
    
# =============================================================================
#   find selected units
# =============================================================================

res = solve_single_knapsack(excess_test, data_test['cost'], 200, method = 'mt1r')
selected_test_idx = np.arange(0, 2000)[res==1]
select_test_score = np.array(test_score_pred[selected_test_idx])
    
# =============================================================================
#   calibrate prediction intervals
# =============================================================================
calib_cost = np.array(data_calib['cost'])
for idx in selected_test_idx:
    reference_id = []
    for ii in range(n_calib):
        # find reference set
        new_excess_test = np.array(excess_test)
        new_cost = np.array(data_test['cost'])
        new_excess_test[idx] = excess_calib[ii]
        new_cost[idx] = calib_cost[ii]
        new_res = solve_single_knapsack(new_excess_test, new_cost, 200, method = 'mt1r')
        if new_res[idx] == 1:
            reference_id.append(ii)
    select_calib_score = calib_score_pred[reference_id]


    # compute prediction intervals

    n_select_calib = len(select_calib_score)
    n_select_test = len(select_test_score)
    if n_select_calib > 0:

        # realized p-values pj(Yj)

        pval = (1+np.sum(test_score_pred[idx] <= select_calib_score)) / (1+len(select_calib_score))
        pval_rand = (np.sum(test_score_pred[idx] < select_calib_score) + (1+np.sum(select_calib_score == test_score_pred[idx])) * np.random.uniform(size=1)[0]) / (1+len(select_calib_score))
        
        # invert p-values to get prediction set for Yj  

        pj_all = [(1 + np.sum(select_calib_score[j] <= select_calib_score)) / (1+len(select_calib_score)) for j in range(len(select_calib_score))]
        pj_all_base = [np.sum(select_calib_score[j] < select_calib_score)  for j in range(len(select_calib_score))]
        pj_all_tie = [ (1+np.sum(select_calib_score==select_calib_score[j]))  for j in range(len(select_calib_score))]
          
        pj_rand_all = (np.array(pj_all_base) + np.array(pj_all_tie) * np.random.uniform(size=1)[0]) / (1+len(select_calib_score))
            
        if np.sum(np.array(pj_rand_all) >= q_ind/10)>0:
            hat_eta_sel_rand = np.max(select_calib_score[np.array(pj_rand_all) >= q_ind/10])
        else:
            hat_eta_sel_rand = 0
        
        if np.sum(np.array(pj_all) >= q_ind/10) > 0:
            hat_eta_sel = np.max(select_calib_score[np.array(pj_all) >= q_ind/10])
        else:
            hat_eta_sel = 0
         
        select_low = test_mu[idx] - hat_eta_sel
        select_upp = test_mu[idx] + hat_eta_sel
        miscover_jomi = 1 * (pval < q_ind/10)
        size_jomi = 2 * hat_eta_sel
        miscover_jomi_rand = 1 * (pval_rand < q_ind/10)
        size_jomi_rand = 2 * hat_eta_sel_rand
        miscover_naive = 1 * (test_score_pred[idx] > hat_eta)
        size_naive = 2 * hat_eta   
        
    else: 
        
        miscover_naive = 1 * (test_score_pred[idx] > hat_eta)
        miscover_jomi = 0
        miscover_jomi_rand = 0
        size_naive = 2 * hat_eta
        size_jomi = np.Inf
        size_jomi_rand = np.Inf 
                
    results = pd.concat([results, pd.DataFrame({"miscover": [miscover_jomi, 
                                                             miscover_jomi_rand, 
                                                             miscover_naive], 
                                               "size": [size_jomi, size_jomi_rand, 
                                                        size_naive],  
                                               "num_select": len(selected_test_idx),
                                               "num_ref": len(reference_id),
                                               "method": ['JOMI', 'JOMI_rand', 'naive'], 
                                               "marginal": 1- q_ind/10, 
                                               'seed': seed})], axis=0)
    
results.to_csv(SAVE_PATH+"/constraint_seed"+str(seed)+"_alpha_"+str(q_ind)+".csv")   
      
