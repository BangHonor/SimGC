# synthetic graph
for reduction_rate in 0.005
    do
    for model in 'GCN' 'SGC' 'SAGE' 'GIN' 'JKNet' 'MLP' 
    do
        python -u SimGC.py --dataset ogbn-products --reduction_rate=${reduction_rate}  --lr_adj=0.01 --lr_feat=0.05 --feat_alpha=10 --smoothness_alpha=0.1 --condensing_loop=3500 --teacher_model=SGC_Multi --model=${model} --seed=1
    done
done
