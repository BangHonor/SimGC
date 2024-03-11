for seed in 1 2 3 4 5
do
    for reduction_rate in 0.25 0.5 1
    do
        python -u SimGC.py --dataset cora --lr_adj=0.001 --lr_feat=0.05 --reduction_rate=${reduction_rate} --teacher_model=SGC --gpu_id=0 --seed=${seed}
    done
done