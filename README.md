# JOMI-paper

This repository contains code for reproducing numerical experiments in the paper "[Confidence on the focal: Conformal prediction with selection-conditional coverage](https://arxiv.org/abs/2403.03868)".



#### File organization 

`/data` contains pre-trained outcomes for drug discovery in Section 5 and health risk prediction in Section 6. 
`/exp-drug` contains scripts for running the drug discovery experiments in Section 5.  
`/exp-health` contains scripts for running the health risk prediction experiments in Section 6.


## 1. Drug discovery experiments 

We provide code for running both drug property prediction (DPP, classification) and drug-target interaction prediction (DTI, regression). 

#### 1.1. Top-K selection 

**Top-K for DPP.** `./exp-drug/topk_dpp.py` is the script for running JOMI for DPP after top-K selection (both lowest and highest ranking) for all choices of K in our paper. It takes as inputs:
- `seed` random seed
- `alpha` nominal miscoverage level (times 10), e.g., 1 for miscoverage rate 0.1
- `output_dir` output directory 
- `rank` what rank to use. `test` for top-K among test sample, `calib` for top-K among calibration sample, and `test_calib` for top-K among both test and calibration data 
- `pred` name for nonconformity score. `bin` for the binary score, and `APS` for the APS score

For example, to run JOMI for DPP after top-K selection among test sample with random seed `1`, nominal miscoverage level `0.1` and the binary score, while storing the output results in `./results/topK_DPP/`, run the following command:

```
python3 topK_dpp.py --seed 1 --alpha 1 --output_dir "./results/topK_DPP/" --rank 'test' --pred 'bin'
```

It then outputs a csv file `topK_test_DPP_bin_seed_1_alpha_1.csv` in the designated folder, where each row stores the empirical miscoverage over selected test points and average prediction set size with experiment parameters at one value of K for either highest or lowest ranking. 

**Top-K for DTI.** `./exp-drug/topk_dti.py` is the script for running JOMI for DTI after top-K selection for all choices of K in our paper (both lowest and highest ranking). It takes as inputs:
- `seed` random seed
- `alpha` nominal miscoverage level (times 10), e.g., 1 for miscoverage rate 0.1
- `output_dir` output directory 
- `rank` what rank to use. `test` for top-K among test sample, `calib` for top-K among calibration sample, and `test_calib` for top-K among both test and calibration data 

It always uses the residual score $V(x,y) = |y-\hat\mu(x)|$. To run JOMI for DTI after top-K selection among both test and calibration sample with random seed `1`, nominal miscoverage level `0.1` while storing the output results in `./results/topK_DTI/`, run the following command:

```
python3 topK_dti.py --seed 1 --alpha 1 --output_dir "./results/topK_DTI/" --rank 'test_calib' 
```

It then outputs a csv file `topK_test_calib_DTI_seed_1_alpha_1.csv` in the designated folder, where each row stores the empirical miscoverage over selected test points and average prediction set size with experiment parameters at one value of K for either highest or lowest ranking. 


#### 1.2. Conformalized selection

**Conformal selection for DPP.** `./exp-drug/confsel_dpp.py` is the script for running JOMI for DPP after conformalized selection at all FDR levels in our paper. It takes as inputs:
- `seed` random seed
- `alpha` nominal miscoverage level (times 10), e.g., 1 for miscoverage rate 0.1
- `output_dir` output directory  
- `pred` name for nonconformity score. `bin` for the binary score, and `APS` for the APS score

To run JOMI for DPP after conformal selection at all FDR levels with random seed `1`, nominal miscoverage level `0.1` and the binary score, while storing the output results in `./results/confsel_DPP/`, run the following command:

```
python3 confsel_dpp.py --seed 1 --alpha 1 --output_dir "./results/confsel_DPP/" --pred 'bin'
```

It then outputs a csv file `confsel_DPP_bin_seed_1_alpha_1.csv` in the designated folder, where each row stores the empirical miscoverage over selected test points and average prediction set size with experiment parameters at one FDR level. 

**Conformal selection for DTI.** `./exp-drug/confsel_dti.py` is the script for running JOMI for DTI after conformalized selection at all FDR levels in our paper. It takes as inputs:
- `seed` random seed
- `alpha` nominal miscoverage level (times 10), e.g., 1 for miscoverage rate 0.1
- `output_dir` output directory  

It always uses the residual score $V(x,y) = |y-\hat\mu(x)|$. To run JOMI for DTI after conformal selection at all FDR levels with random seed `101`, nominal miscoverage level `0.3` while storing the output results in `./results/confsel_DTI/`, run the following command:

```
python3 confsel_dti.py --seed 101 --alpha 3 --output_dir "./results/confsel_DTI/" 
```


It then outputs a csv file `confsel_DTI_seed_101_alpha_3.csv` in the designated folder, where each row stores the empirical miscoverage over selected test points and average prediction set size with experiment parameters at one FDR level. 

#### 1.3. Selection under constraints

**Constrained selection for DPP.** `./exp-drug/constraint_dpp.py` is the script for running JOMI, randomized JOMI, and vanilla conformal prediction for DPP after selection with budget constraints. It takes as inputs:
- `seed` random seed
- `alpha` nominal miscoverage level (times 10), e.g., 1 for miscoverage rate 0.1
- `output_dir` output directory  
- `pred` name for nonconformity score. `bin` for the binary score, and `APS` for the APS score

To run JOMI for DPP after constrained selection with random seed `531`, nominal miscoverage level `0.4` and the APS score, while storing the output results in `./results/cons_DPP/`, run the following command:

```
python3 constraint_dpp.py --seed 531 --alpha 4 --output_dir "./results/cons_DPP/" --pred 'APS'
```

It then outputs a csv file `opt_DPP_seed_531_alpha_4.csv` in the designated folder, where each row contains the miscover indicator and prediction set size for one selected test unit, together with experiment parameters.

**Constrained selection for DTI.** `./exp-drug/confsel_dti.py` is the script for running JOMI, randomized JOMI, and vanilla conformal prediction for DTI after selection with budget constraints. It takes as inputs:
- `seed` random seed
- `alpha` nominal miscoverage level (times 10), e.g., 1 for miscoverage rate 0.1
- `output_dir` output directory  

It always uses the residual score $V(x,y) = |y-\hat\mu(x)|$. To run JOMI for DTI after selection under budget constraints with random seed `312`, nominal miscoverage level `0.2` while storing the output results in `./results/cons_DTI/`, run the following command:

```
python3 confsel_dti.py --seed 312 --alpha 2 --output_dir "./results/cons_DTI/" 
```

It then outputs a csv file `opt_DTI_seed_312_alpha_2.csv` in the designated folder, where each row contains the miscover indicator and prediction set size for one selected test unit, together with experiment parameters. 

## 2. Health risk prediction experiments

#### 2.1. Selection after optimization

`./exp-health/constraint.py` is the script for running JOMI, randomized JOMI, and vanilla conformal prediction for health risk prediction, when test units selected by Knapsack optimization, and outputs a csv file with results. It takes as inputs:
- `seed` random seed
- `alpha` nominal miscoverage level (times 10), e.g., 1 for miscoverage rate 0.1
- `output_dir` output directory  

It always uses $V(x,y) = |y-\hat\mu(x)|$ as the nonconformity score. To run the script at random seed `100`, miscoverage level `0.1`, and write the output in `./results/opt/`, navigate to the folder `/exp-health` and run the command: 

```
python3 constraint.py --seed 100 --alpha 1 --output_dir "./results/opt/"
```

It then outputs a csv file `constraint_seed100_alpha_1.csv` in the designated folder, where each row contains the miscover indicator and prediction set size for one selected test unit and one procedure, together with experiment parameters.

#### 2.2. Selection based on PI length

`./exp-health/length.py` runs JOMI, randomized JOMI, and vanilla conformal prediction for health risk prediction, when test units whose preliminary conformal prediction sets based on score $V(x,y) = |y-\hat\mu(x)|/\hat\sigma(x)$ are shorter than 5. It uses the same score for constructing JOMI prediction sets. It takes the same inputs as Section 2.1. To run the script at random seed `100`, miscoverage level `0.1`, and write the output in `./results/length/`, navigate to the folder `/exp-health` and run the command: 


```
python3 length.py --seed 100 --alpha 1 --output_dir "./results/length/"
```

It then outputs a csv file `length_seed_100_alpha_1.csv` in the designated folder, where each row contains the miscover indicator and prediction size for one test unit using one of the procedures, as well as experiment parameters and some intermediate outcomes. 

#### 2.3. Selection based on PI upper bound

`./exp-health/lower.py` runs JOMI, randomized JOMI, and vanilla conformal prediction for health risk prediction, when test units whose preliminary conformal prediction sets based on score $V(x,y) = |y-\hat\mu(x)|/\hat\sigma(x)$ have an upper bound below 6. It uses the same score for constructing JOMI prediction sets. It takes the same inputs as Section 2.1. To run the script at random seed `100`, miscoverage level `0.1`, and write the output in `./results/lower/`, navigate to the folder `/exp-health` and run the command: 


```
python3 lower.py --seed 100 --alpha 1 --output_dir "./results/lower/"
```

It then outputs a csv file `lower_seed_100_alpha_1.csv` in the designated folder, where each row contains the miscover indicator and prediction size for one test unit using one of the procedures, as well as experiment parameters and some intermediate outcomes. 

#### Reference
```
@article{jin2024confidence,
  title={Confidence on the focal: Conformal prediction with selection-conditional coverage},
  author={Jin, Ying and Ren, Zhimei},
  journal={arXiv preprint arXiv:2403.03868},
  year={2024}
}
```
