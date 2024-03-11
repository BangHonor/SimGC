# for nlayers in 2 3 4
# do
#     for hidden in 64 128 256 512
#     do
#         for dropout in 0 0.3 0.5
#         do
#             for activation in 'sigmoid' 'tanh' 'relu' 'softplus' 'leakyreLU' 'relu6' 'elu'
#             do
#                 python -u SimGC.py --dataset reddit --nlayers=${nlayers} --reduction_rate=0.002 --hidden=${hidden} --dropout=${dropout} --activation=${activation} --seed=1 
#             done
#         done
#     done
# done

for nlayers in 2 3 4
do
    for hidden in 128 256 512
    do
        for dropout in 0 0.3 0.5
        do
            for activation in 'sigmoid' 'tanh' 'relu' 'softplus' 'leakyreLU' 'elu'
            do
                python -u nas_inductive.py --dataset reddit --nlayers=${nlayers} --hidden=${hidden} --dropout=${dropout} --activation=${activation} --gpu_id=0 --seed=1 
            done
        done
    done
done