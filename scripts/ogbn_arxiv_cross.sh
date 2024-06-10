# synthetic graph
for seed in 1 2 3 4 5
    do
    for model in 'GCN' 'SGC' 'SAGE' 'GIN' 'JKNet' 'MLP' 
    do
        python -u SimGC.py --dataset ogbn-arxiv --reduction_rate=0.01  --lr_adj=0.01 --lr_feat=0.05 --feat_alpha=10 --smoothness_alpha=0.1 --condensing_loop=1500 --teacher_model=SGC_Multi --model=${model} --seed=${seed} 
    done
done
