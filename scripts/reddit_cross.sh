# synthetic graph
for seed in 1 2 3 4 5
do
    for model in 'GCN' 'SGC' 'SAGE' 'GIN' 'JKNet' 'MLP' 
    do
        python -u SimGC_inductive.py --dataset reddit --reduction_rate=0.002  --lr_adj=0.01 --lr_feat=0.05 --feat_alpha=100 --smoothness_alpha=0.1 --condensing_loop=2500 --teacher_model=SGC --model=${model} --gpu_id=1 --seed=${seed} 
    done
done
