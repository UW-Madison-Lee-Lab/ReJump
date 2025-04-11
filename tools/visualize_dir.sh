
#!/usr/bin/env bash

BASE_DIR="results/deepseek-ai-deepseek-reasoner"

for parquet_file in $(find "$BASE_DIR" -type f -name "test_default.parquet"); do
    echo "Processing: $parquet_file"
    python tools/visualize.py --input="$parquet_file"
done
