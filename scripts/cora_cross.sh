# synthetic graph
for seed in 1 2 3 4 5
    do
    for model in 'GCN' 'SGC' 'SAGE' 'GIN' 'JKNet' 'MLP' 
    do
        python -u SimGC.py --dataset cora --reduction_rate=0.5  --lr_adj=0.001 --lr_feat=0.005 --feat_alpha=10 --smoothness_alpha=0.1 --condensing_loop=1500 --teacher_model=SGC_Multi  --model=${model} --gpu_id=1 --seed=${seed}
    done
done
