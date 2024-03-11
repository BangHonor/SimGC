# synthetic graph
for seed in 1 2 3 4 5
    do
    for model in 'GCN' 'SGC' 'SAGE' 'GIN' 'JKNet' 'MLP' 
    do
        python -u SimGC.py --dataset ogbn-arxiv --reduction_rate=0.01 --teacher_model=SGC --model=${model} --seed=${seed} 
    done

    for model in 'GCN' 'SGC' 'SAGE' 'GIN' 'JKNet' 'MLP' 
    do
        python -u SimGC.py --dataset ogbn-arxiv --reduction_rate=0.01 --teacher_model=GCN --model=${model} --seed=${seed} 
    done
done
