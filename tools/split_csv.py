import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Create pivot tables from regression metrics data.")
    parser.add_argument("input_csv", help="Path to the input CSV file")
    parser.add_argument("output_csv", help="Path to the output CSV file")
    args = parser.parse_args()

    # Read the CSV file from the specified path
    df = pd.read_csv(args.input_csv)

    # Split the 'experiment' column into 'model' and 'test_item'
    df[['model', 'test_item']] = df['experiment'].str.rsplit('_', n=1, expand=True)

    # Identify metric columns by excluding known columns
    metrics = [col for col in df.columns if col not in ['experiment', 'model', 'test_item']]

    pivot_tables = {}
    for metric in metrics:
        if metric in df.columns:
            # Create a pivot table for each metric using mean values
            pivot_table = df.pivot_table(index='test_item', columns='model', values=metric, aggfunc='mean')
            pivot_tables[metric] = pivot_table
        else:
            print(f"Column '{metric}' not found, skipping.")

    # Write pivot tables to the specified output CSV file
    with open(args.output_csv, "w", encoding="utf-8") as f:
        for metric, table in pivot_tables.items():
            f.write(f"Table for {metric}:\n")
            table.to_csv(f)
            f.write("\n")

if __name__ == "__main__":
    main()
