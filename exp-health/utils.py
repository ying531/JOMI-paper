import numpy as np
import random 
import pandas as pd
import sys
import os 
import argparse

def binary_score_select(y, hat_mu, score, if_test, M=100., c=None):
    if if_test: # test score V(x,c)
        if score == 'residual': # V(x,y) = y - hat_mu
            return c - hat_mu
        if score == 'clip':
            return c - hat_mu
    else: # calibration score V(x,y)
        if score == 'residual': # V(x,y) = y - hat_mu
            return y - hat_mu
        if score == 'clip': # V(x,y) = M I(y>c) + c I(y<=c) - hat_mu
            return M * 1. * (y>c) + c * 1. * (y<=c) - hat_mu

def binary_score(y, hat_mu, score):
    if score == 'bin':
        return y * (1-hat_mu) + (1-y) * hat_mu
    if score == 'APS':
        return y * (hat_mu * 1 * (hat_mu >= 0.5) + 1 * (hat_mu < 0.5)) + (1 - y) * ((1-hat_mu) * 1 * (hat_mu < 0.5) + 1 * (hat_mu >= 0.5))
    
    
def regress_score_select(y, hat_mu, score, if_test, M=100., c=None):
    if if_test: # test score V(x,c)
        if score == 'residual': # V(x,y) = y - hat_mu
            return c - hat_mu
        if score == 'clip':
            return c - hat_mu
    else: # calibration score V(x,y)
        if score == 'residual': # V(x,y) = y - hat_mu
            return y - hat_mu
        if score == 'clip': # V(x,y) = M I(y>c) + c I(y<=c) - hat_mu
            return M * 1. * (y>c) + c * 1. * (y<=c) - hat_mu
        
def regress_score(y, hat_mu, score):
    if score == 'residual':
        return np.absolute(y - hat_mu)

# compute T for conformalized selection
def compute_T_df(calib_score_low, test_score, n_calib, n_test):
    all_df = pd.DataFrame({"S": list(calib_score_low) + list(test_score), 
                           "if_test": [0]*len(calib_score_low) + [1] * n_test})
    all_df = all_df.sort_values(by = 'S')
    all_df['F'] = (1+np.cumsum(1-all_df['if_test'])) / np.maximum(1, np.cumsum(all_df['if_test']))
    all_df['F'] = all_df['F'] * n_test / (1+n_calib)
    return all_df

# compute T1 used in constructing the reference set after conformalized selection
def compute_T_other_df(calib_score_low, test_score_loo, n_calib, n_test):
    all_df = pd.DataFrame({"S": list(calib_score_low) + list(test_score_loo), 
                           "if_test": [0]*len(calib_score_low) + [1] * len(test_score_loo)})
    all_df = all_df.sort_values(by = 'S')
    all_df['F'] = (1+np.cumsum(1-all_df['if_test'])) / (1+ np.cumsum(all_df['if_test']))
    all_df['F'] = all_df['F'] * n_test / (1+n_calib)
    return all_df


# compute T2 used in constructing the reference set after conformalized selection
def compute_T_other_2_df(calib_score_low, test_score_loo, n_calib, n_test):
    all_df = pd.DataFrame({"S": list(calib_score_low) + list(test_score_loo), 
                           "if_test": [0]*len(calib_score_low) + [1] * len(test_score_loo)})
    all_df = all_df.sort_values(by = 'S')
    all_df['F'] = np.cumsum(1-all_df['if_test'])  / (1 + np.cumsum(all_df['if_test']))
    all_df['F'] = all_df['F'] * n_test / (1+n_calib)
    return all_df


def compute_left_interval(mu, eta, cj):
    if eta == np.Inf:
        return np.array([-np.Inf, cj])
    if cj <= mu - eta:
        return None
    if cj <= mu + eta:
        return np.array([mu-eta, cj])
    if cj > mu + eta:
        return np.array([mu-eta, mu+eta])
    
def compute_right_interval(mu, eta, cj):
    if eta == np.Inf:
        return np.array([cj, np.Inf])
    if cj <= mu - eta:
        return np.array([mu-eta, mu+eta])
    if cj <= mu + eta:
        return np.array([cj, mu+eta])
    if cj > mu+eta:
        return None
    