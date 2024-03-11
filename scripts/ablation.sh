
# for alignment in 0 1
# do
#     for smoothness in 0 1
#     do
#         python -u SimGC.py --dataset ogbn-arxiv --model=GCN --alignment=$((alignment)) --smoothness=$((smoothness))
#     done
# done


for alignment in 0 1
do
    for smoothness in 0 1
    do
        python -u SimGC.py --dataset ogbn-products --alignment=$((alignment)) --smoothness=$((smoothness))
    done
done

# for alignment in 0 1
# do
#     for smoothness in 0 1
#     do
#         python -u SimGC.py --dataset cora --reduction_rate=1 --threshold=0.05 --model=GCN --alignment=$((alignment)) --smoothness=$((smoothness))
#     done
# done

# for alignment in 0 1
# do
#     for smoothness in 0 1
#     do
#         python -u SimGC_inductive.py --dataset reddit --reduction_rate=0.002 --model=GCN --alignment=$((alignment)) --smoothness=$((smoothness))
#     done
# done

# for alignment in 0 1
# do
#     for smoothness in 0 1
#     do
#         python -u temp.py --dataset reddit2 --reduction_rate=0.002 --model=GCN --alignment=$((alignment)) --smoothness=$((smoothness))
#     done
# done

# for alignment in 0 1
# do
#     for smoothness in 0 1
#     do
#         python -u SimGC.py --dataset cora-arxiv --reduction_rate=0.005 --condensing_loop=2500 --teacher_model=GCN --model=GCN --alignment=$((alignment)) --smoothness=$((smoothness))
#     done
# done

