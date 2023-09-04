# A pytorch implementation for paper "Criterion-based Heterogeneous Collaborative Filtering for Multi-behavior Implicit Recommendation" 

## REQUIREMENTS
1. pytorch 1.4
2. loguru
3. scipy
4. sklearn

## DATASETS
[BeiBei][https://github.com/chenchongthu/EHCF]
[Taobao][https://github.com/chenchongthu/EHCF]

## EXPERIMENTS

Here we use the GMF as the CF basic model.

How to train Beibei : python search.py 0 [GPU_ID]

How to train Taobao : python search.py 1 [GPU_ID]

Our model will be restored in the checkpoints.
