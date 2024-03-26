import numpy as np
import random 
import pandas as pd
import sys
import os 
import argparse
from utils import regress_score, regress_score_select
parser = argparse.ArgumentParser('')
 
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--alpha', type=int, default=1, 
    choices = [1,2,3,4,5,6,7,8,9], help='nominal miscoverage level (times 10)') 
parser.add_argument('--output_dir', type=str, 
    required=True, help='output directory for saving results') 
parser.add_argument('--rank', type=str, default='test', 
    choices = ['test', 'calib', 'test_calib'], 
    help='test for top-K among test data, calib for top-K among calibration data, test_calib for top-K among mixed data') 

args = parser.parse_args()
seed = args.seed
SAVE_PATH = args.output_dir
q_ind = args.alpha
rank_method = args.rank 
score_pred = 'residual'

if not os.path.exists(SAVE_PATH):
  os.makedirs(SAVE_PATH)
  

raw_calib = pd.read_csv("../results/DTI/CNN_transformer_calib_"+str(seed % 10 + 1)+".csv")
raw_test = pd.read_csv("../results/DTI/CNN_transformer_test_"+str(seed % 10 + 1)+".csv")
raw_test.columns = raw_calib.columns
data = pd.concat([raw_calib, raw_test], axis=0).iloc[:,1:raw_calib.shape[1]]
data = data.rename({"calib_pred": "pred", "calib_true": "y"}, axis = "columns")


results = pd.DataFrame() 
 
np.random.seed(seed)
n = data.shape[0]
reind = np.random.permutation(n)

data_calib = data.iloc[reind[0:int(n/2)]].reset_index().iloc[:,1:data.shape[1]]
data_test = data.iloc[reind[int(n/2):n]].reset_index().iloc[:,1:data.shape[1]]

# conformal prediction set 
calib_y = data_calib['y']
calib_mu = data_calib['pred'] + np.random.normal(size=len(calib_y)) * 0.0001
calib_score = regress_score(calib_y, calib_mu, score = score_pred)
n_calib = len(calib_y)  
hat_eta = np.quantile(np.array(list(calib_score)+[np.Inf]), 
                      1-q_ind/10, 
                      interpolation='higher') 
    
# test scores
test_y = np.array(data_test['y'])
test_mu = np.array(data_test['pred']) + np.random.normal(size=len(test_y)) * 0.0001
test_score = regress_score(test_y, test_mu, score = score_pred)
n_test = len(test_mu)

# =============================================================================
# run the procedure for all K's
# =============================================================================

Ks = [20, 100, 1000, 2000, 5000, 10000]
    
    # =========================================== 
    # find selected test units and reference set
    # =========================================== 
    
    for K in Ks:

        if rank_method == 'test':
            select_calib_score = np.array(calib_score[calib_mu >= np.sort(test_mu)[n_test-K-1]] )
            select_test_score = np.array(test_score[test_mu >= np.sort(test_mu)[n_test-K]]) 
            select_test_mu = test_mu[test_mu >= np.sort(test_mu)[n_test-K]] 
            
        if rank_method == 'test_calib': 
            TT = np.sort(np.array(list(calib_mu) + list(test_mu)))[n_test+n_calib-K]
            select_calib_score = np.array(calib_score[calib_mu >= TT])             
            select_test_score = np.array(test_score[test_mu >= TT]) 
            select_test_mu = test_mu[test_mu >= TT] 
        
        if rank_method == 'calib':
            select_calib_score = np.array(calib_score[calib_mu > np.sort(calib_mu)[n_calib-K]] )
            select_test_score = np.array(test_score[test_mu > np.sort(calib_mu)[n_calib-K]])   
            select_test_mu = test_mu[test_mu > np.sort(calib_mu)[n_calib-K]]
            select_test_y = test_y[test_mu > np.sort(calib_mu)[n_calib-K]]  
            
            
        # compute prediction intervals
        n_select_calib = len(select_calib_score)
        n_select_test = len(select_test_score)

        if n_select_calib > 0:

            # compute selective p-values and prediction sets

            # realized p-values pj(Yj)
            pval = []
            pval_rand = []
            for i in range(n_select_test):
                pval = pval + [(1+np.sum(select_test_score[i] <= select_calib_score)) / (1+len(select_calib_score))]
                pval_rand = pval_rand + [(np.sum(select_test_score[i] < select_calib_score) + (1+np.sum(select_calib_score == select_test_score[i])) * np.random.uniform(size=1)[0]) / (1+len(select_calib_score))]
            
            # invert p-values to get prediction set for Yj 

            hat_eta_sel_list = []
            hat_eta_sel_rand_list = []
            pj_all = [(1 + np.sum(select_calib_score[j] <= select_calib_score)) / (1+len(select_calib_score)) for j in range(len(select_calib_score))]
            pj_all_base = [np.sum(select_calib_score[j] < select_calib_score)  for j in range(len(select_calib_score))]
            pj_all_tie = [ (1+np.sum(select_calib_score==select_calib_score[j]))  for j in range(len(select_calib_score))]
            
            for i in range(n_select_test):
                pj_rand_all = (np.array(pj_all_base) + np.array(pj_all_tie) * np.random.uniform(size=1)[0]) / (1+len(select_calib_score))
                
                if np.sum(np.array(pj_rand_all) >= q_ind/10)>0:
                    hat_eta_sel_rand_list += [np.max(select_calib_score[np.array(pj_rand_all) >= q_ind/10])] 
                else:
                    hat_eta_sel_rand_list += [0]
                
                if np.sum(np.array(pj_all) >= q_ind/10) > 0:
                    hat_eta_sel_list += [np.max(select_calib_score[np.array(pj_all) >= q_ind/10])]
                else:
                    hat_eta_sel_list += [0]
             
            select_low = select_test_mu - np.array(hat_eta_sel_list)
            select_upp = select_test_mu + np.array(hat_eta_sel_list)
            miscover_jomi = np.mean(np.array(pval) < q_ind/10)
            size_jomi = 2 * np.mean(np.array(hat_eta_sel_list))
            miscover_jomi_rand = np.mean(np.array(pval_rand) < q_ind/10)
            size_jomi_rand = 2 * np.mean(np.array(hat_eta_sel_rand_list))
            miscover_naive = np.mean(select_test_score > hat_eta)
            size_naive = 2 * hat_eta   
            
        else: 
            miscover_naive = np.mean(select_test_score > hat_eta)
            miscover_jomi = 0
            miscover_jomi_rand = 0
            size_naive = 2 * hat_eta
            size_jomi = np.Inf
            size_jomi_rand = np.Inf 
            
        results = pd.concat([results, 
                             pd.DataFrame({"miscover":
                                           [miscover_naive, miscover_jomi, miscover_jomi_rand], 
                                           "size": 
                                           [size_naive, size_jomi, size_jomi_rand], 
                                           "num_select": n_select_test,
                                           "method": ['Vanilla_CP', "JOMI", "JOMI_rand"], 
                                                   "marginal": 1- q_ind/10,
                                                   "subseed": seed,
                                                   "K": K, 
                                                   "order": "highest"})], axis=0) 


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
            select_test_mu = test_mu[test_mu <= np.sort(test_mu)[K-1]] 
            
        if rank_method == 'test_calib': 
            TT = np.sort(np.array(list(calib_mu) + list(test_mu)))[K-1]
            select_calib_score = np.array(calib_score[calib_mu <= TT])             
            select_test_score = np.array(test_score[test_mu <= TT]) 
            select_test_mu = test_mu[test_mu <= TT] 
            
        if rank_method == 'calib':
            select_calib_score = np.array(calib_score[calib_mu <= np.sort(calib_mu)[K-1]] )
            select_test_score = np.array(test_score[test_mu <= np.sort(calib_mu)[K-1]])   
            select_test_mu = test_mu[test_mu <= np.sort(calib_mu)[K-1]]  
            
        n_select_calib = len(select_calib_score)
        n_select_test = len(select_test_score)

        if n_select_calib > 0:

            # realized p-values pj(Yj)
            pval = []
            pval_rand = []
            for i in range(n_select_test):
                pval = pval + [(1+np.sum(select_test_score[i] <= select_calib_score)) / (1+len(select_calib_score))]
                pval_rand = pval_rand + [(np.sum(select_test_score[i] < select_calib_score) + (1+np.sum(select_calib_score == select_test_score[i])) * np.random.uniform(size=1)[0]) / (1+len(select_calib_score))]
            
            # invert p-values to get prediction set for Yj 
            hat_eta_sel_list = []
            hat_eta_sel_rand_list = []
            pj_all = [(1 + np.sum(select_calib_score[j] <= select_calib_score)) / (1+len(select_calib_score)) for j in range(len(select_calib_score))]
            pj_all_base = [np.sum(select_calib_score[j] < select_calib_score)  for j in range(len(select_calib_score))]
            pj_all_tie = [ (1+np.sum(select_calib_score==select_calib_score[j]))  for j in range(len(select_calib_score))]
            
            for i in range(n_select_test):
                pj_rand_all = (np.array(pj_all_base) + np.array(pj_all_tie) * np.random.uniform(size=1)[0]) / (1+len(select_calib_score))
                
                if np.sum(np.array(pj_rand_all) >= q_ind/10)>0:
                    hat_eta_sel_rand_list += [np.max(select_calib_score[np.array(pj_rand_all) >= q_ind/10])] 
                else:
                    hat_eta_sel_rand_list += [0]
                
                if np.sum(np.array(pj_all) >= q_ind/10) > 0:
                    hat_eta_sel_list += [np.max(select_calib_score[np.array(pj_all) >= q_ind/10])]
                else:
                    hat_eta_sel_list += [0]
             
            select_low = select_test_mu - np.array(hat_eta_sel_list)
            select_upp = select_test_mu + np.array(hat_eta_sel_list)
            miscover_jomi = np.mean(np.array(pval) < q_ind/10)
            size_jomi = 2 * np.mean(np.array(hat_eta_sel_list))
            miscover_jomi_rand = np.mean(np.array(pval_rand) < q_ind/10)
            size_jomi_rand = 2 * np.mean(np.array(hat_eta_sel_rand_list))
            miscover_naive = np.mean(select_test_score > hat_eta)
            size_naive = 2 * hat_eta   
            
             
            
        else: 
            miscover_naive = np.mean(select_test_score > hat_eta)
            miscover_jomi = 0
            miscover_jomi_rand = 0
            size_naive = 2 * hat_eta
            size_jomi = np.Inf
            size_jomi_rand = np.Inf 
            
        results = pd.concat([results, 
                             pd.DataFrame({"miscover":
                                           [miscover_naive, miscover_jomi, miscover_jomi_rand], 
                                           "size": 
                                           [size_naive, size_jomi, size_jomi_rand], 
                                           "num_select": n_select_test,
                                           "method": ['Vanilla_CP', "JOMI", "JOMI_rand"], 
                                                   "marginal": 1- q_ind/10,
                                                   "seed": seed,
                                                   "K": K, 
                                                   "order": "lowest"})], axis=0) 
         
     
results.to_csv(SAVE_PATH+"/topK_" + rank_method + "_DTI_seed_"+str(seed)+"_alpha_"+str(q_ind)+".csv")
 