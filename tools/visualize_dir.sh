
#!/usr/bin/env bash

BASE_DIR="results/10query"


python tools/split_csv.py $BASE_DIR/regression_metrics.csv $BASE_DIR/regression_output.csv
python tools/split_csv.py $BASE_DIR/classification_metrics.csv $BASE_DIR/classification_output.csv

