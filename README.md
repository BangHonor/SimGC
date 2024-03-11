# Simple Graph Condensation
Official codebase for paper Simple Graph Condensation.  This codebase is based on the open-source Pytorch Geometric framework.

## Overview

**Abstart:** The burdensome training costs on large-scale graphs have aroused significant interest in graph condensation, which involves tuning Graph Neural Networks (GNNs) on a small condensed graph for use on the large-scale original graph. Existing methods primarily focus on aligning key metrics between the condensed and original graphs, such as gradients, output distribution and trajectories of GNNs, \lsy{yielding} satisfactory performance on downstream tasks.
However, these complex metrics necessitate intricate computations and can potentially disrupt the optimization process of the condensation graph, making the condensation process highly demanding and unstable.
Motivated by the recent success of simplified models in various fields, we propose a simplified approach to metric alignment in graph condensation, aiming to reduce unnecessary complexity inherited from GNNs.
In our approach, we eliminate external parameters and exclusively retain the target condensed graph during the condensation process.
Following the hierarchical aggregation principles of GNNs, we introduce the Simple Graph Condensation (SimGC) framework, which aligns the condensed graph with the original graph from the input layer to the prediction layer, guided by a pre-trained Simple Graph Convolution (SGC) model on the original graph. As a result, both graphs possess the similar capability to train GNNs.
This straightforward yet effective strategy achieves a significant speedup of up to 10 times compared to existing graph condensation methods while performing on par with state-of-the-art baselines.
Comprehensive experiments conducted on seven benchmark datasets demonstrate the effectiveness of SimGC in prediction accuracy, condensation time, and generalization capability.

![SimGC_framework 图标]([https://github.com/BangHonor/DisCo/blob/main/SimGC_framework.png](https://github.com/BangHonor/SimGC/blob/main/SimGC_framework.png))

## Requirements
See requirments.txt file for more information about how to install the dependencies.

## Run the Code
For transductive setting, please run the following command:
```
python -u SimGC.py --dataset ogbn-arxiv --reduction_rate=${reduction_rate} --teacher_model=SGC --gpu_id=1 --seed=1
```


For inductive setting, please run the following command:
```
python -u SimGC_inductive.py --dataset reddit --reduction_rate=${reduction_rate} --teacher_model=SGC --model=GCN --gpu_id=1 --seed=1 
```


## Reproduce
Please follow the instructions below to replicate the results in the paper.

Run to reproduce the results of Table 2  in "scripts/cora.sh",  "scripts/pubmed.sh", "scripts/ogbn-arxiv.sh", "scripts/ogbn-products.sh", "scripts/reddit.sh", "scripts/reddit2.sh".

Run to reproduce the results of Table 6 in "scripts/nas_transductive.sh", "scripts/nas_inductive.sh".

Run to reproduce the results of Table 7 in "scripts/kd_transductive.sh" . 


## Contact
Please feel free to contact me via email (xiaozhb@zju.edu.cn) if you are interested in my research :)
