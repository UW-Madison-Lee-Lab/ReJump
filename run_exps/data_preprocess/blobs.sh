for n in 10 50 100 200
do
    python /Users/cychomatica/Documents/code/liftr/examples/data_preprocess/blobs.py \
    --template_type=no_reasoning \
    --num_samples=1000 \
    --n_features=2 \
    --centers=3 \
    --cluster_std=1.0 \
    --test_ratio=0.2 \
    --n_shot=$n
done
