import pandas as pd

# 读入 CSV 文件（假设文件名为 "results/regression_metrics.csv"）
df = pd.read_csv("results/classification_metrics.csv")

# 解析 experiment 字段，将字符串从最后一个下划线处分为 model 与 test_item
df[['model', 'test_item']] = df['experiment'].str.rsplit('_', n=1, expand=True)

metrics = [col for col in df.columns if col not in ['experiment', 'model', 'test_item']]

pivot_tables = {}
for metric in metrics:
    if metric in df.columns:
        # 使用 pivot_table，可以通过 aggfunc='mean' 对重复的记录进行平均
        pivot_table = df.pivot_table(index='test_item', columns='model', values=metric, aggfunc='mean')
        pivot_tables[metric] = pivot_table
    else:
        print(f"Column '{metric}' not found, skipping.")

# 将所有构造的数据透视表写入到一个 CSV 文件中
output_csv = "results/classification_output.csv"
with open(output_csv, "w", encoding="utf-8") as f:
    for metric, table in pivot_tables.items():
        f.write(f"Table for {metric}:\n")
        table.to_csv(f)
        f.write("\n")
