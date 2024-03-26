import numpy as np
import random 
import pandas as pd
import sys
import os 
import argparse
from utils import binary_score_select, binary_score
parser = argparse.ArgumentParser('')

parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--q', type=int, default=1, 
    choices = [1,2,3,4,5,6,7,8,9], help='nominal miscoverage level (times 10)') 
parser.add_argument('--output_dir', type=str, required=True, 
    help='output directory for saving results')  
parser.add_argument('--pred', type=str, default='bin', 
    help='nonconformity score function, bin for binary score, APS for APS score')

args = parser.parse_args()
seed = args.seed
SAVE_PATH = args.output_dir
q_ind = args.q  
score_pred = args.pred
  

if not os.path.exists(SAVE_PATH):
  os.makedirs(SAVE_PATH)
  


data = pd.read_csv("../results/DPP/DPP_"+str(seed % 10 + 1)+".csv")
data = pd.DataFrame({"pred": list(data['calib_pred']) + list(data['test_pred']),
                     "y": list(data['calib_true']) + list(data['test_true'])})

np.random.seed(1)
data = data.iloc[np.random.choice(data.shape[0], 
                                  5000, replace=False)]

data['cost'] = np.exp(data['pred']) + np.abs(np.sin(data['pred'])) + np.random.exponential(size = data.shape[0]) - 1 + np.random.uniform(size = data.shape[0])

 
results = pd.DataFrame() 
 
np.random.seed(seed)
n = data.shape[0]
reind = np.random.permutation(n)

data_calib = data.iloc[reind[0:int(n/2)]].copy(deep=True)
data_calib = data_calib.reset_index(drop=True)
data_test = data.iloc[reind[int(n/2):n]].copy(deep=True)
data_test = data_test.reset_index(drop=True)

# conformal prediction set 
calib_y = data_calib['y']
calib_mu = data_calib['pred'] + np.random.normal(size=len(calib_y)) * 0.0001
calib_score = binary_score(calib_y, calib_mu, score_pred)  
 
n_calib = len(calib_y) 
hat_eta = np.quantile(np.array(list(calib_score)+[np.Inf]), 
                      1-q_ind/10, 
                      interpolation='higher')

# test scores
test_y = np.array(data_test['y'])
test_mu = np.array(data_test['pred']) + np.random.normal(size=len(test_y)) * 0.0001
n_test = len(test_mu)
test_score = binary_score(test_y, test_mu, score_pred)  
test_score_1 = binary_score(1, test_mu, score_pred)  
test_score_0 = binary_score(0, test_mu, score_pred)  
    

set_1_in = 1 * (test_score_1 <= hat_eta)
set_0_in = 1 * (test_score_0 <= hat_eta) 
set_test = set_1_in + set_1_in * set_0_in - (1-set_1_in) * (1-set_0_in) 
    
# =============================================================================
#     # find original selection set
# =============================================================================
test_sort = data_test.copy(deep=True)
test_sort['idx'] = np.arange(0, data_test.shape[0])
test_sort = test_sort.sort_values(by = 'pred', ascending = False)
test_sort['accum'] = np.cumsum(test_sort['cost'])
test_selected = test_sort[test_sort['accum'] < 500]
select_test_score = np.array(test_score[test_selected['idx']])
    
# =============================================================================
#     # find reference calibration points
# =============================================================================
ref_nums = []
sel_qtls = []
pval = []
pval_0 = []
pval_1 = []
pval_rand = []
pval_rand_0 = []
pval_rand_1 = []


for idx in test_selected['idx']:
    
    # =============================================================================
    #         # find reference points
    # =============================================================================
    reference_id = [ii for ii in range(data_calib.shape[0]) if np.sum(data_test[data_test['pred'] >= data_calib['pred'].iloc[ii]]['cost']) - test_selected['cost'].loc[idx] * 1 * (test_selected['pred'].loc[idx] > data_calib['pred'].iloc[ii])  + data_calib.iloc[ii]['cost'] < 500]  
    
    # reference scores 
    select_calib_score = np.array(calib_score[np.array(reference_id)]) 
    n_select_calib = len(select_calib_score)  
    
    if n_select_calib > 0:
        
        # =============================================================================
        # # compute p-values
        # =============================================================================
        # non-randomized
        pval = (1+np.sum(test_score[idx] <= select_calib_score)) / (1+len(select_calib_score))
        
        pval_0 = (1+np.sum(test_score_0[idx] <= select_calib_score)) / (1+len(select_calib_score))
        pval_1 = (1+np.sum(test_score_1[idx] <= select_calib_score)) / (1+len(select_calib_score))
        
        # randomized
        pval_rand = (np.sum(test_score[idx] < select_calib_score) + (1+np.sum(select_calib_score == test_score[idx])) * np.random.uniform(size=1)[0]) / (1+len(select_calib_score))
        
        Uj = np.random.uniform(size=1)[0]
        pval_rand_0 = (np.sum(test_score_0[idx] < select_calib_score) + (1+np.sum(select_calib_score == test_score_0[idx])) * Uj) / (1+len(select_calib_score))
        pval_rand_1 = (np.sum(test_score_1[idx] < select_calib_score) + (1+np.sum(select_calib_score == test_score_1[idx])) * Uj) / (1+len(select_calib_score))
        
         
        # =============================================================================
        # # construct prediction sets
        # =============================================================================
             
        select_set_1_in = 1 * (np.array(pval_1) >= q_ind/10) 
        select_set_0_in = 1 * (np.array(pval_0) >= q_ind/10)   
            
        # else: # randomize
        select_set_1_in_rand = 1 * (np.array(pval_rand_1) >= q_ind/10) 
        select_set_0_in_rand = 1 * (np.array(pval_rand_0) >= q_ind/10) 
        
        naive_miscover = 1*(set_test[idx] < 2)*(test_y[idx]!=set_test[idx])
        jomi_miscover = 1 * (pval < q_ind/10)
        jomi_rand_miscover = 1 * (pval_rand<q_ind/10)
        naive_size = set_1_in[idx] + set_0_in[idx]
        jomi_size = 1 * (select_set_1_in) + 1 * (select_set_0_in)
        jomi_rand_size = 1 * (select_set_1_in_rand) + 1 * (select_set_0_in_rand)
        
    else: 
        naive_miscover = 0
        jomi_miscover = 0 
        jomi_rand_miscover = 0
        naive_size = 2
        jomi_size = 2
        jomi_rand_size = 2  


    results = pd.concat([results, 
                         pd.DataFrame({"miscover":
                                       [naive_miscover, jomi_miscover, jomi_rand_miscover], 
                                       "size": 
                                       [naive_size, jomi_size, jomi_rand_size], 
                                       "num_reference": n_select_calib,
                                       "method": ['Vanilla_CP', "JOMI", "JOMI_rand"], 
                                       "marginal": 1- q_ind/10,
                                       "seed": seed, 
                                       "pred_score": score_pred})], axis=0)
         
         
        

    results.to_csv(SAVE_PATH+"/opt_DPP_"+score_pred+"_seed_"+str(seed)+"_q_"+str(q_ind)+".csv")   


