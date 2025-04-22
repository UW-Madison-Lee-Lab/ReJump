import json
from environment import root_dir
from constants import get_result_dir, supported_datasets
import pandas as pd
from utils import load_json
import matplotlib.pyplot as plt
import tiktoken
import seaborn as sns
import argparse

def normalize(text):
    return ' '.join(text.split())

def get_model_text(idx, model_type_json, model_text):
    llm_extracted_json = json.loads(model_text["llm_analysis_extracted_json"].iloc[idx])
    full_reasoning = normalize(model_text["responses"].iloc[idx][0])
    
    for i in range(1, len(llm_extracted_json)):
        model_parsed_txt = normalize(llm_extracted_json[i]["rule_original_text"].split("...")[0])
        try:
            p1, p2 = full_reasoning.split(model_parsed_txt)
            model_type_json[i-1]["text"] = p1
            full_reasoning = model_parsed_txt + p2
        except ValueError:
            model_type_json[i-1]["text"] = ""
        except IndexError:
            print(idx, i, len(llm_extracted_json), len(model_type_json))
            print("Skip this one.")

    return model_type_json

def count_tokens(text):
    try:
        # Using cl100k_base tokenizer (similar to what Qwen uses)
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except:
        # Fallback: approximate token count if tiktoken is not available
        return len(text.split())

def get_df(
    dataset_name,
    model_name,
    response_length,
):
    result_dir = get_result_dir(
        dataset_name = dataset_name,
        model_name = model_name,
        shot = 50,
        template_type = "reasoning_api",
        response_length = response_length,
        num_samples = 500,
        feature_noise = supported_datasets[dataset_name]["feature_noise"],
        label_noise = supported_datasets[dataset_name]["label_noise"],
        data_mode = "default",
        n_query = 10,
    )
    model_funcs = load_json(f"{result_dir}/test_default_gemini_analysis_llm_analysis.json")
    model_text = pd.read_parquet(f"{result_dir}/test_default_gemini_analysis.parquet")

    all_texts, all_accs, all_token_counts, all_cumulative_counts = [], [], [], []
    for i in range(model_funcs["metadata"]["processed_samples"]):
        prompt_idx = model_funcs["samples"][i]["index"]
        models_type_json = model_funcs["samples"][i]["model_evaluation_table"]
        model_type_json = get_model_text(prompt_idx, models_type_json, model_text)
        texts = [model["text"] for model in model_type_json if "text" in model]
        accs = [model["accuracy"] for model in model_type_json if "text" in model]
        token_counts = [count_tokens(text) for text in texts]
        cumulative_token_counts = [sum(token_counts[:j+1]) for j in range(len(token_counts))]
        
        all_texts.extend(texts)
        all_accs.extend(accs)
        all_token_counts.extend(token_counts)
        all_cumulative_counts.extend(cumulative_token_counts)
    
    df = pd.DataFrame({
        "acc": all_accs,
        "token_count": all_cumulative_counts
    })
    return df

def draw_acc_vs_token_count(dfs, labels, figure_name):
    plt.figure(figsize=(10, 6))
    for df, label in zip(dfs, labels):
        sns.regplot(
            x="token_count", 
            y="acc", 
            data=df, 
            label=label,
            order=2,
            scatter=False,
            line_kws={'linewidth': 2}
        )
    plt.xlabel('Token Count')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Token Count')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{root_dir}/figures/{figure_name}.pdf')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, nargs="+", required=True)
    parser.add_argument("--response_length", type=int, required=True)
    args = parser.parse_args()

    dfs = []
    labels = []
    model_str = ""
    for model_name in args.model_name:
        df = get_df(args.dataset_name, model_name, args.response_length)
        dfs.append(df)
        labels.append(model_name)
        model_str += f"{model_name.replace('/', '-')}_"
    draw_acc_vs_token_count(dfs, labels, f"{args.dataset_name}_{model_str}_acc_vs_token_count")