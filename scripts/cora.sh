for seed in 1 2 3 4 5
do
    for reduction_rate in 0.25 0.5 1
    do
        python -u SimGC.py --dataset cora --reduction_rate=${reduction_rate}  --lr_adj=0.001 --lr_feat=0.005 --feat_alpha=10 --smoothness_alpha=0.1 --condensing_loop=1500 --teacher_model=SGC_Multi --gpu_id=0 --seed=${seed}
    done
done
