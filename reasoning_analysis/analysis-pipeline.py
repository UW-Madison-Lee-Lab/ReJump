# find all test_default_gemini_analysis.parquet files under a specific directory(default:/home/szhang967/liftr/multi-query-results)
#for each file, run python /home/szhang967/liftr/reasoning_analysis/llm_analysis.py --input <file_path>
#then the llm_analyis output file is saved in the same directory as the input file, name
#then run python /home/szhang967/liftr/reasoning_analysis/plot_model_accuracy.py --input llm_analyis_output_file_path

import os
import subprocess
import argparse
from pathlib import Path

def get_data_type(file_path):
    data_type_dict = {
        "circles": "classification",
        "blobs": "classification",
        "moons": "classification",
        "linear": "classification",

        "l1normreg": "regression",
        "quadreg": "regression",
        "linreg": "regression",
        "cosreg": "regression",
        "pwreg": "regression",
    }
    for key, value in data_type_dict.items():
        if key in file_path:
            return value
    raise NotImplementedError(f"Unknown data type for file: {file_path}")

def run_analysis_pipeline(base_dir="/home/szhang967/liftr/multi-query-results"):
    # Find all test_default_gemini_analysis.parquet files
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == "test_default_gemini_analysis.parquet":
                input_file = os.path.join(root, file)
                data_type = get_data_type(input_file)

                print(f"Processing file: {input_file}")
                # if data_type != "classification":
                #     continue
                # if "claude" not in input_file:
                #     continue
                # if "circles" not in input_file:
                #     continue
                
                # # Run llm_analysis.py
                # analysis_script = "/home/szhang967/liftr/reasoning_analysis/llm_analysis.py"
                # subprocess.run(["python", analysis_script, "--input", input_file, "--data_type", data_type], check=True)

                # Run llm_logical_graph_analysis.py
                logical_graph_input_file = input_file.replace("test_default_gemini_analysis.parquet", "test_default_gemini_analysis_logical_graph.parquet")
                # logical_graph_analysis_script = "/home/szhang967/liftr/reasoning_analysis/llm_logical_graph_analysis.py"
                # subprocess.run(["python", logical_graph_analysis_script, "--input", logical_graph_input_file], check=True)
                
                # # Get the output file path (same directory as input)
                # output_dir = os.path.dirname(input_file)
                # analysis_output = os.path.join(output_dir,  f"{file.split('.')[0]}" + "_llm_analysis.json")
                
                
                # # Run plot_model_accuracy.py
                # if os.path.exists(analysis_output):
                #     plot_script = "/home/szhang967/liftr/reasoning_analysis/plot_model_accuracy.py"
                #     subprocess.run(["python", plot_script, "--input", analysis_output, "--data_type", data_type], check=True)
                # else: 
                #     print(f"Warning: Analysis output file not found at {analysis_output}")

def parse_args():
    parser = argparse.ArgumentParser(description='Run analysis pipeline on Gemini test results.')
    parser.add_argument('--base-dir', 
                      type=str, 
                      default="/home/szhang967/liftr/multi-query-results",
                      help='Base directory to search for test_default_gemini_analysis.parquet files')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_analysis_pipeline(args.base_dir)

