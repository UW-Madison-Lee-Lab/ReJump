for n in 10 50 100 200
do
    python ./examples/data_preprocess/linear.py \
    --template_type=no_reasoning \
    --num_samples=1000 \
    --n_shot=$n
done
