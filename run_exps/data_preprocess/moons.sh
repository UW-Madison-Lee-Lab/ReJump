for n in 10 50 100 200
do
    python /Users/cychomatica/Documents/code/liftr/examples/data_preprocess/moons.py \
    --template_type=no_reasoning \
    --num_samples=1000 \
    --n_shot=$n
done
