# for reduction_rate in 0.00025 0.0001
# do
#     python -u SimGC.py --dataset ogbn-products --reduction_rate=${reduction_rate} --seed=1 
# done

for seed in 1 2 3 4 5
do
    for reduction_rate in 0.0025 0.005 0.01
    do
        python -u SimGC.py --dataset ogbn-products --reduction_rate=${reduction_rate} --threshold=0.05  --seed=${seed} --teacher_model='SGC' --validation_model='GCN' 
    done
done