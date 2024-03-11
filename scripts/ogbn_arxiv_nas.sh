# synthetic graph
# for nlayers in 2 3 4
# do
#     for hidden in 128 256 512
#     do
#         for dropout in 0 0.3 0.5
#         do
#             for activation in 'sigmoid' 'tanh' 'relu' 'softplus' 'leakyreLU' 'elu'
#             do
#                 python -u SimGC.py --dataset ogbn-arxiv --nlayers=${nlayers} --hidden=${hidden} --dropout=${dropout} --activation=${activation} --teacher_model=SGC --model=GCN --seed=1 
#             done
#         done
#     done
# done

# original graph
for nlayers in 2 3 4
do
    for hidden in 128 256 512
    do
        for dropout in 0 0.3 0.5
        do
            for activation in 'sigmoid' 'tanh' 'relu' 'softplus' 'leakyreLU' 'elu'
            do
                python -u nas_transductive.py --dataset ogbn-arxiv --nlayers=${nlayers} --hidden=${hidden} --dropout=${dropout} --activation=${activation} --seed=1 
            done
        done
    done
done