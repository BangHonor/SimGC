for reduction_rate in 0.0005 0.001
do
    python -u SimGC_inductive.py --dataset reddit2 --reduction_rate=${reduction_rate} --teacher_model=SGC --model=GCN --gpu_id=0 --seed=1
done

for seed in 2 3 4 5
do
    for reduction_rate in 0.0005 0.001 0.002
    do
        python -u SimGC_inductive.py --dataset reddit2 --reduction_rate=${reduction_rate} --teacher_model=SGC --model=GCN --gpu_id=0 --seed=${seed}
    done
done