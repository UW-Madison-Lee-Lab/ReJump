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
        
        "expreg": "regression",
        "l1normreg": "regression",
        "quadreg": "regression",
        "linreg": "regression",
        "cosreg": "regression",
        "pwreg": "regression",

        "gsm8k": "unknown",
    }
    for key, value in data_type_dict.items():
        if key in file_path:
            return value
    raise NotImplementedError(f"Unknown data type for file: {file_path}")

def get_model_name(file_path):
    if "deepseek-ai-deepseek-reasoner" in file_path:
        return "deepseek-reasoner"
    elif "claude-claude-3-7-sonnet-20250219-thinking" in file_path:
        return "claude-3-7-sonnet"
    else:
        return "unknown"

def run_logical_graph_analysis(base_dir="/home/szhang967/liftr/multi-query-results"):
    # Path to analysis script

    ###debug
    base_dir = "/home/szhang967/liftr/test_sample_graph_blobs"
     ###
    analysis_script = "/home/szhang967/liftr/reasoning_analysis/analyze_responses.py"
    instruction_file = "/home/szhang967/liftr/reasoning_analysis/cognitive_process_graph_prompt.txt"
    
    # Process all test_default.parquet files
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == "test_default.parquet":
                input_file = os.path.join(root, file)
                
                # Skip files that aren't from the models we're interested in
                model_name = get_model_name(input_file)
            #     if model_name == "unknown":
            #         continue
                # if "gsm8k" not in input_file:
                #     continue
                # if "deepseek" not in input_file:
                #     continue
                
                # Get data type for organization/logging
                data_type = get_data_type(input_file)
                
                print(f"Processing file: {input_file}")
                print(f"Model: {model_name}, Data type: {data_type}")
                
                # Run analysis
                cmd = [
                    "python", analysis_script,
                    "--input", input_file,
                    "--instruction", instruction_file,
                    "--llm", "gemini",
                    "--temperature", "0",
                    "--max_tokens", "35000",
                    "--output_suffix", "_cognitive_process_graph",
                    "--field_of_interests", "input+reasonings",
                    # "--debug"
                ]
                try:
                    subprocess.run(cmd, check=True)
                    print(f"Successfully processed {input_file}")
                except subprocess.CalledProcessError as e:
                    print(f"Error processing {input_file}: {e}")
                
                gemini_analysis_output_file = input_file.replace("test_default.parquet", "test_default_gemini_analysis_cognitive_process_graph.parquet")
                logical_graph_analysis_script = "/home/szhang967/liftr/reasoning_analysis/llm_logical_graph_analysis.py"
                subprocess.run(["python", logical_graph_analysis_script, "--input", gemini_analysis_output_file], check=True)
                

                
def parse_args():
    parser = argparse.ArgumentParser(description='Run logical graph analysis pipeline on model results.')
    parser.add_argument('--base-dir', 
                      type=str, 
                      default="/home/szhang967/liftr/multi-query-results",
                      help='Base directory to search for test_default.parquet files')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_logical_graph_analysis(args.base_dir) 