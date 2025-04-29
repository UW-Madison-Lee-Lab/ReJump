
#!/usr/bin/env bash

BASE_DIR="results/"

for parquet_file in $(find "$BASE_DIR" -type f -name "test_default.parquet"); do
    echo "Processing: $parquet_file"
    python tools/visualize.py --input="$parquet_file" --output-csv=$BASE_DIR
done
