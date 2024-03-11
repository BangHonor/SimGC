# synthetic graph
for reduction_rate in 0.005
    do
    for model in 'GCN' 'SGC' 'SAGE' 'GIN' 'JKNet' 'MLP' 
    do
        python -u SimGC.py --dataset ogbn-products --reduction_rate=${reduction_rate} --teacher_model=GCN --model=${model} --seed=1
    done
done
