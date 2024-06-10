# for reduction_rate in 0.00025 0.0001
# do
#     python -u SimGC.py --dataset ogbn-products --reduction_rate=${reduction_rate} --seed=1 
# done

for seed in 1 2 3 4 5
do
    for reduction_rate in 0.0025 0.005 0.01
    do
        python -u SimGC.py --dataset ogbn-products --reduction_rate=${reduction_rate}  --lr_adj=0.01 --lr_feat=0.05 --feat_alpha=10 --smoothness_alpha=0.1 --condensing_loop=1500 --teacher_model=SGC_Multi  --seed=${seed} --validation_model='GCN' 
    done
done
