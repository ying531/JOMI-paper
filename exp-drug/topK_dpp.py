import numpy as np
import random 
import pandas as pd
import sys
import os 
import argparse
from utils import binary_score_select, binary_score
parser = argparse.ArgumentParser('')

parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--alpha', type=int, default=1, 
    choices = [1,2,3,4,5,6,7,8,9], help='nominal miscoverage level (times 10)') 
parser.add_argument('--output_dir', type=str, 
    required=True, help='output directory for saving results') 
parser.add_argument('--rank', type=str, default='test', 
    choices = ['test', 'calib', 'test_calib'], 
    help='test for top-K among test data, calib for top-K among calibration data, test_calib for top-K among mixed data')
parser.add_argument('--pred', type=str, default='bin', 
    choices = ['bin', 'APS'], 
    help='nonconformity score for prediction interval, bin for binary score, APS for the APS score')

args = parser.parse_args()
seed = args.seed
SAVE_PATH = args.output_dir
q_ind = args.alpha
rank_method = args.rank 
score_pred = args.pred 

if not os.path.exists(SAVE_PATH):
  os.makedirs(SAVE_PATH)
  
 
data = pd.read_csv("../data/DPP/DPP_"+str(seed % 10 + 1)+".csv")
data = pd.DataFrame({"pred": list(data['calib_pred']) + list(data['test_pred']),
                     "y": list(data['calib_true']) + list(data['test_true'])})

results = pd.DataFrame()  
np.random.seed(seed)
n = data.shape[0]
reind = np.random.permutation(n)

data_calib = data.iloc[reind[0:int(n/2)]].reset_index().iloc[:,1:3]
data_test = data.iloc[reind[int(n/2):n]].reset_index().iloc[:,1:3]

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
 
Ks = [20, 100, 1000, 2000, 5000, 10000, 15000]  

# =============================================================================
# run the procedure for all K's
# =============================================================================

for K in Ks:

    # =========================================== 
    # find selected test units and reference set
    # =========================================== 
    
    if rank_method == 'test':
        select_calib_score = np.array(calib_score[calib_mu >= np.sort(test_mu)[n_test-K-1]] )
        select_test_score = np.array(test_score[test_mu >= np.sort(test_mu)[n_test-K]]) 
        select_test_score_1 = test_score_1[test_mu >= np.sort(test_mu)[n_test-K]]
        select_test_score_0 = test_score_0[test_mu >= np.sort(test_mu)[n_test-K]]
        select_test_y = test_y[test_mu >= np.sort(test_mu)[n_test-K]] 
        select_test_set_naive = set_test[test_mu >= np.sort(test_mu)[n_test-K]]

    if rank_method == 'test_calib': 
        TT = np.sort(np.array(list(calib_mu) + list(test_mu)))[n_test+n_calib-K]
        select_calib_score = np.array(calib_score[calib_mu >= TT])             
        select_test_score = np.array(test_score[test_mu >= TT]) 
        select_test_score_1 = test_score_1[test_mu >= TT]
        select_test_score_0 = test_score_0[test_mu >= TT]
        select_test_y = test_y[test_mu>=TT] 
        select_test_set_naive = set_test[test_mu >= TT]

    if rank_method == 'calib':
        select_calib_score = np.array(calib_score[calib_mu > np.sort(calib_mu)[n_calib-K]] )
        select_test_score = np.array(test_score[test_mu > np.sort(calib_mu)[n_calib-K]]) 
        select_test_score_1 = test_score_1[test_mu > np.sort(calib_mu)[n_calib-K]]
        select_test_score_0 = test_score_0[test_mu > np.sort(calib_mu)[n_calib-K]]
        select_test_y = test_y[test_mu > np.sort(calib_mu)[n_calib-K]] 
        select_test_set_naive = set_test[test_mu > np.sort(calib_mu)[n_calib-K]]
         
     
    n_select_calib = len(select_calib_score)
    n_select_test = len(select_test_score)
     
    
    if n_select_calib > 0: 

        # compute selective p-values and prediction sets

        pval = []
        pval_rand = []
        for i in range(n_select_test):
            pval = pval + [(1+np.sum(select_test_score[i] <= select_calib_score)) / (1+len(select_calib_score))]
            pval_rand = pval_rand + [(np.sum(select_test_score[i] < select_calib_score) + (1+np.sum(select_calib_score == select_test_score[i])) * np.random.uniform(size=1)[0]) / (1+len(select_calib_score))]
         
        pval_0 = []
        pval_1 = []
        pval_rand_0 = []
        pval_rand_1 = []

        for i in range(n_select_test): 
            # p-values and randomized p-values to find prediction sets
            Uj = np.random.uniform(size=1)[0]
            pval_0 = pval_0 + [(1+np.sum(select_test_score_0[i] <= select_calib_score)) / (1+len(select_calib_score))]
            pval_1 = pval_1 + [(1+np.sum(select_test_score_1[i] <= select_calib_score)) / (1+len(select_calib_score))]
            pval_rand_0 = pval_rand_0 + [(np.sum(select_test_score_0[i] < select_calib_score) + (1+np.sum(select_calib_score == select_test_score[i])) * Uj) / (1+len(select_calib_score))]
            pval_rand_1 = pval_rand_1 + [(np.sum(select_test_score_1[i] < select_calib_score) + (1+np.sum(select_calib_score == select_test_score[i])) * Uj) / (1+len(select_calib_score))]
         
        select_set_1_in = 1 * (np.array(pval_1) >= q_ind/10) 
        select_set_0_in = 1 * (np.array(pval_0) >= q_ind/10)   
            
        # randomized prediction set
        select_set_1_in_rand = 1 * (np.array(pval_rand_1) >= q_ind/10) 
        select_set_0_in_rand = 1 * (np.array(pval_rand_0) >= q_ind/10)  
        
        naive_miscover = np.mean(1*(select_test_set_naive<2)*(select_test_y!=select_test_set_naive))
        jomi_miscover = np.mean(np.array(pval)< q_ind/10)
        jomi_rand_miscover = np.mean(np.array(pval_rand)<q_ind/10)
        naive_size = np.mean(select_test_score_1 <= hat_eta) + np.mean(select_test_score_0 <= hat_eta)
        jomi_size = np.mean(select_set_1_in) + np.mean(select_set_0_in)
        jomi_rand_size = np.mean(select_set_1_in_rand) + np.mean(select_set_0_in_rand)
        
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
                                       "num_select": n_select_test,
                                       "method": ['Vanilla_CP', "JOMI", "JOMI_rand"], 
                                               "marginal": 1- q_ind/10,
                                               "subseed": seed,
                                               "K": K, 
                                               "order": "highest", 
                                               "pred_score": score_pred})], axis=0)

# =============================================================================
# run the procedure for all K's (top-K lowest rank)
# =============================================================================
     
for K in Ks: 

    # =========================================== 
    # find selected test units and reference set
    # =========================================== 

    if rank_method == 'test':
        select_calib_score = np.array(calib_score[calib_mu <= np.sort(test_mu)[K]] )
        select_test_score = np.array(test_score[test_mu <= np.sort(test_mu)[K-1]]) 
        select_test_score_1 = test_score_1[test_mu <= np.sort(test_mu)[K-1]]
        select_test_score_0 = test_score_0[test_mu <= np.sort(test_mu)[K-1]] 
        select_test_y = test_y[test_mu <= np.sort(test_mu)[K-1]] 
        select_test_set_naive = set_test[test_mu <= np.sort(test_mu)[K-1]]

    if rank_method == 'test_calib': 
        TT = np.sort(np.array(list(calib_mu) + list(test_mu)))[K-1]
        select_calib_score = np.array(calib_score[calib_mu <= TT])             
        select_test_score = np.array(test_score[test_mu <= TT]) 
        select_test_score_1 = test_score_1[test_mu <= TT]
        select_test_score_0 = test_score_0[test_mu <= TT] 
        select_test_y = test_y[test_mu<=TT] 
        select_test_set_naive = set_test[test_mu <= TT]

    if rank_method == 'calib':
        select_calib_score = np.array(calib_score[calib_mu <= np.sort(calib_mu)[K-1]] )
        select_test_score = np.array(test_score[test_mu <= np.sort(calib_mu)[K-1]]) 
        select_test_score_1 = test_score_1[test_mu <= np.sort(calib_mu)[K-1]]
        select_test_score_0 = test_score_0[test_mu <= np.sort(calib_mu)[K-1]]
        select_test_y = test_y[test_mu <= np.sort(calib_mu)[K-1]] 
        select_test_set_naive = set_test[test_mu <= np.sort(calib_mu)[K-1]]
         
        
        
    n_select_calib = len(select_calib_score)
    n_select_test = len(select_test_score)
    
    
    if n_select_calib > 0:
        
        pval = []
        pval_rand = []
        for i in range(n_select_test):
            pval = pval + [(1+np.sum(select_test_score[i] <= select_calib_score)) / (1+len(select_calib_score))]
            pval_rand = pval_rand + [(np.sum(select_test_score[i] < select_calib_score) + (1+np.sum(select_calib_score == select_test_score[i])) * np.random.uniform(size=1)[0]) / (1+len(select_calib_score))]
         
        pval_0 = []
        pval_1 = []
        pval_rand_0 = []
        pval_rand_1 = []
        for i in range(n_select_test): 
            Uj = np.random.uniform(size=1)[0]
            pval_0 = pval_0 + [(1+np.sum(select_test_score_0[i] <= select_calib_score)) / (1+len(select_calib_score))]
            pval_1 = pval_1 + [(1+np.sum(select_test_score_1[i] <= select_calib_score)) / (1+len(select_calib_score))]
            pval_rand_0 = pval_rand_0 + [(np.sum(select_test_score_0[i] < select_calib_score) + (1+np.sum(select_calib_score == select_test_score[i])) * Uj) / (1+len(select_calib_score))]
            pval_rand_1 = pval_rand_1 + [(np.sum(select_test_score_1[i] < select_calib_score) + (1+np.sum(select_calib_score == select_test_score[i])) * Uj) / (1+len(select_calib_score))]
         
        select_set_1_in = 1 * (np.array(pval_1) >= q_ind/10) 
        select_set_0_in = 1 * (np.array(pval_0) >= q_ind/10)   
            
        # randomize
        select_set_1_in_rand = 1 * (np.array(pval_rand_1) >= q_ind/10) 
        select_set_0_in_rand = 1 * (np.array(pval_rand_0) >= q_ind/10)  
        
        naive_miscover = np.mean(1*(select_test_set_naive<2)*(select_test_y!=select_test_set_naive))
        jomi_miscover = np.mean(np.array(pval)< q_ind/10)
        jomi_rand_miscover = np.mean(np.array(pval_rand)<q_ind/10)
        naive_size = np.mean(select_test_score_1 <= hat_eta) + np.mean(select_test_score_0 <= hat_eta)
        jomi_size = np.mean(select_set_1_in) + np.mean(select_set_0_in)
        jomi_rand_size = np.mean(select_set_1_in_rand) + np.mean(select_set_0_in_rand)
            
        
    else: 
        naive_miscover = 0
        jomi_miscover = 0 
        jomi_rand_miscover = 0
        naive_size = 2
        jomi_size = 2
        jomi_rand_size = 2  


    results = pd.concat([results, 
                         pd.DataFrame({"method": ['Vanilla_CP', "JOMI", "JOMI_rand"], 
                                       "miscover":
                                       [naive_miscover, jomi_miscover, jomi_rand_miscover], 
                                       "size": 
                                       [naive_size, jomi_size, jomi_rand_size], 
                                       "num_select": n_select_test, 
                                       "marginal": 1- q_ind/10,
                                       "seed": seed,
                                       "K": K, 
                                       "order": "lowest", 
                                       "pred_score": score_pred})], axis=0)
         
         
results.to_csv(SAVE_PATH+"/topK_" + rank_method + "_DPP_"+score_pred+"_seed_"+str(seed)+"_alpha_"+str(q_ind)+".csv")  



