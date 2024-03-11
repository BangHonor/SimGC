# synthetic graph
for seed in 1 2 3 4 5
    do
    for model in 'GCN' 'SGC' 'SAGE' 'GIN' 'JKNet' 'MLP' 
    do
        python -u SimGC_inductive.py --dataset reddit2 --reduction_rate=0.002 --teacher_model=SGC --model=${model} --gpu_id=1 --seed=${seed} 
    done
done
