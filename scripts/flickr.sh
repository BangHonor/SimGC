for seed in 1 2 3 4 5
do
    for reduction_rate in 0.001 0.005 0.01
    do
        python -u SimGC_induct.py --dataset flickr --reduction_rate=${reduction_rate} --teacher_model=SGC --gpu_id=1 --seed=${seed}
    done
done