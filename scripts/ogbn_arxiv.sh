for seed in 1 2 3 4 5
do
    for reduction_rate in 0.001 0.005 0.01
    do
        python -u SimGC_transductive.py --dataset ogbn-arxiv --reduction_rate=${reduction_rate}  --lr_adj=0.01 --lr_feat=0.05 --feat_alpha=10 --smoothness_alpha=0.1 --condensing_loop=1500 --teacher_model=SGC --gpu_id=1 --seed=${seed}
    done
done
