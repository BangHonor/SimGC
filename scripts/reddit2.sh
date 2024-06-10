for seed in 1 2 3 4 5
do
    for reduction_rate in 0.0005 0.001 0.002
    do
        python -u SimGC_inductive.py --dataset reddit2 --reduction_rate=${reduction_rate}  --lr_adj=0.01 --lr_feat=0.05 --feat_alpha=100 --smoothness_alpha=0.1 --condensing_loop=2500 --teacher_model=SGC --model=GCN --gpu_id=0 --seed=${seed}
    done
done
