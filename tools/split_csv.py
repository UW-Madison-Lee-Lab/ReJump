import pandas as pd

df = pd.read_csv("results/regression_metrics.csv")

df[['model', 'test_item']] = df['experiment'].str.rsplit('_', n=1, expand=True)

metrics = [col for col in df.columns if col not in ['experiment', 'model', 'test_item']]

pivot_tables = {}
for metric in metrics:
    if metric in df.columns:
        pivot_table = df.pivot_table(index='test_item', columns='model', values=metric, aggfunc='mean')
        pivot_tables[metric] = pivot_table
    else:
        print(f"Column '{metric}' not found, skipping.")

output_csv = "results/regression_output.csv"
with open(output_csv, "w", encoding="utf-8") as f:
    for metric, table in pivot_tables.items():
        f.write(f"Table for {metric}:\n")
        table.to_csv(f)
        f.write("\n")
