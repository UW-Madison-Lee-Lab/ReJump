import os
import subprocess
import argparse
from pathlib import Path

# google_api_key_list =[
    # yzeng58
# GEMINI_API_KEY = "AIzaSyARTR4pSoM8hmIIMEg85OMHD1T9KgaGwV4"
# zycbb
# GEMINI_API_KEY = "AIzaSyDa6IZiztMRZl_j2_em_MjruiCyZs9vEFs"
# shuibai
# GEMINI_API_KEY = "AIzaSyA0vPcRshKiv5fftazBAbxdIHvtLyHCCiE"
# Jungtaek
# GEMINI_API_KEY = "AIzaSyBgM3S40tAiRJ5J1f-jx8xgecBmbelnPXg"
# yuchenzeng.1998
# GEMINI_API_KEY = "AIzaSyDIAq1UMSGx1-svNob46Rt616JF0UHW3VY"
# shutong
# GEMINI_API_KEY = "AIzaSyCCnSBSjD1CgtighzPQyx03HZWvrVSWTHM"
# lynnix
# GEMINI_API_KEY = "AIzaSyBUhgp-FBViNj8VTxM3Tw8gXJsARgyx-dc"
# ziqian
# GEMINI_API_KEY = "AIzaSyD8PQnKv6AYN_oUVmlNKG8dviEr6mO_J_0"
# ]
#concvert the above info into a list with the name as comments
gemini_api_key_list = [
    # "AIzaSyBgM3S40tAiRJ5J1f-jx8xgecBmbelnPXg",
    "AIzaSyDIAq1UMSGx1-svNob46Rt616JF0UHW3VY",
    "AIzaSyCCnSBSjD1CgtighzPQyx03HZWvrVSWTHM",
    "AIzaSyBUhgp-FBViNj8VTxM3Tw8gXJsARgyx-dc",
    "AIzaSyD8PQnKv6AYN_oUVmlNKG8dviEr6mO_J_0",
    "AIzaSyARTR4pSoM8hmIIMEg85OMHD1T9KgaGwV4",
    "AIzaSyDa6IZiztMRZl_j2_em_MjruiCyZs9vEFs",
    "AIzaSyA0vPcRshKiv5fftazBAbxdIHvtLyHCCiE",
    
]#use these keys by turns
# def get_data_type(file_path):
#     data_type_dict = {
#         "circles": "classification",
#         "blobs": "classification",
#         "moons": "classification",
#         "linear": "classification",
        
#         "expreg": "regression",
#         "l1normreg": "regression",
#         "quadreg": "regression",
#         "linreg": "regression",
#         "cosreg": "regression",
#         "pwreg": "regression",

#         "gsm8k": "unknown",
#         "math500": "unknown",
#     }
#     for key, value in data_type_dict.items():
#         if key in file_path:
#             return value
#     raise NotImplementedError(f"Unknown data type for file: {file_path}")

# def get_model_name(file_path):
#     if "deepseek-ai-deepseek-reasoner" in file_path:
#         return "deepseek-reasoner"
#     elif "claude-claude-3-7-sonnet-20250219-thinking" in file_path:
#         return "claude-3-7-sonnet"
#     else:
#         return "unknown"

def run_logical_graph_analysis(base_dir):
    # Path to analysis script

    ###debug
    # base_dir = "/home/szhang967/liftr/test_sample_graph_blobs"
    # base_dir = "/home/szhang967/liftr/results"
    # base_dir = "/staging/szhang967/liftr/results/openai-gpt-4o/math500_0_shot_1_query_standard_api_reslen_404_nsamples_500_noise_None_flip_rate_0.0_mode_default/global_step_0"
    # base_dir = '/staging/szhang967/results'
     ###
    analysis_script = "/home/szhang967/liftr/reasoning_analysis/analyze_responses.py"
    instruction_file = "/home/szhang967/liftr/reasoning_analysis/cognitive_process_graph_prompt.txt"
    
    model_name_list = [
        # "results_temperature_0/deepseek-ai-DeepSeek-R1-Distill-Qwen-7B",
        # "results_temperature_0/Qwen-Qwen2.5-7B-Instruct",

        # "results_temperature_0/meta-llama-Llama-3.1-8B-Instruct",
        # "results_temperature_0/deepseek-ai-DeepSeek-R1-Distill-Llama-8B",

        # 'results_temperature_0/openrouter-qwen-qwq-32b',

        # "results_temperature_0/Qwen-Qwen2.5-3B-Instruct",

        # 'results_temperature_0/openrouter-microsoft-phi-4',
        "results_temperature_0/xai-grok-3-mini-beta"
    ]
    
    # Process all test_default.parquet files
    key_idx = 0
    for root, dirs, files in os.walk(base_dir):
        for idx, file in enumerate(files):
            key = gemini_api_key_list[key_idx % len(gemini_api_key_list)]
            os.environ["GEMINI_API_KEY"] = key
            key_idx += 1

            if file == "test_default.parquet":
                input_file = os.path.join(root, file)
                if not any(model_name in input_file for model_name in model_name_list):
                    continue
                print(f"Processing file: {input_file}")
                
                # Run analysis
                # import pdb; pdb.set_trace()
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
                os.environ["GEMINI_API_KEY"] = ""

                
def parse_args():
    parser = argparse.ArgumentParser(description='Run logical graph analysis pipeline on model results.')
    parser.add_argument('--base-dir', 
                      type=str, 
                    #   default="/home/szhang967/liftr/multi-query-results",
                      default = '/staging/szhang967/results_temperature_0',
                      help='Base directory to search for test_default.parquet files')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_logical_graph_analysis(args.base_dir) 