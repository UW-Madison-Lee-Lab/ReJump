import json
import os
import argparse
import re
import environment
import numpy as np
from openai import OpenAI
import time
from tqdm import tqdm

def compute_centroids(raw_input):

    X, Y = [], []
    prompt = raw_input['prompt'][0]['content']
    examples = prompt.split('\n')
    examples = [example for example in examples if example.startswith('Features:')]
    for example in examples:
        x, y = example.split(', Label: ')
        x = x.split('Features: ')[-1].strip().split(',')
        x = [float(i.strip()) for i in x]
        y = int(y.strip())
        X.append(x)
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)

    centroids = []
    for i in range(3):
        centroids.append(X[Y == i].mean(axis=0))
    centroids = np.array(centroids)

    return X.tolist(), Y.tolist(), centroids.tolist()
    
def read_outputs(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON format in line: {line.strip()}")
    return data

SYSTEM_PROMPT = '''
You will be given a student's solution for a classification task. Please help me analyze the student's reasoning process. 

Firstly, tell me what kind of algorithm (knn, nearest centroid, rule-based, or other) the student used to solve this task. 

Secondly, figure out the arguments used for the algorithm. For KNN, please find the value of k used and the k nearest neighbors, and record them in your final response. For nearest centroid, if the student proposed the estimated centroids, please find them out and record them in your final response. For rule-based, specify the rule used to classify data points. For other algorithms, you can just skip this part. 

Thirdly, please evaluate the student's reasoning process. For KNN: if the student finds out k nearest neighbors, verify if they are the real nearest neighbors. For nearest centroid classifier, compare the estimated centroids with the true centroids and compute the MSE loss. For rule-based, verify if the proposed rule is rigorously applied. For other algorithms, you can just skip this part.

Please ensure your response strictly follows the given JSON format:
```json
{
    "algorithm": <algorithm used by the student, e.g., KNN or Nearest Centroid>,
    "knn": {
        "k": <value of k for KNN, if applicable, otherwise None>,
        "nearest neighbors":  <a 2d list as [[nn_0_0, nn_0_1, ...], [nn_1_0, nn_1_1, ...], ...] representing the k nearest neighbors, if applicable, otherwise None>,
        "correctness": <true if the student finds the real nearest neighbors and false otherwise, if applicable, otherwise None>
    }

    "ncc": 
    {
        "centroids": <a 2d list as [[label_0_centroid_0, label_0_centroid_1, ...], [label_1_centroid_0, label_1_centroid_1, ...], ...] representing the estimated centroids for all Nearest Centroid, if applicable, otherwise None>,
        "correctness": <true if the student finds the real centroids and false otherwise, if applicable, otherwise None>,
    }
    "rule-based": {
        "rule": <the rule proposed by the student, if applicable, otherwise None>,
        "correctness": <true if the student's rule-based reasoning is correct and false otherwise, if applicable, otherwise None>
    }
    "final_answer": <the final label given by the student's reasoning, if applicable, otherwise None>
}
```
Make sure there are no comments or explanations in the response. 
'''

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='openai/gpt-4o', choices=['deepseek-ai/deepseek-chat', 'deepseek-ai/deepseek-reasoner', 'openai/gpt-4'], help='Model name')
    args = parser.parse_args()

    file_path = '/Users/cychomatica/Documents/code/liftr/deepseek-ai-deepseek-reasoner/blobs_50_shot_base_reslen_3046_nsamples_500_noise_1.0_flip_rate_0.0/global_step_0/test.json'
    data = read_outputs(file_path)
    
    # model = LLMAPI(api_key=environment.OPENAI_API_KEY, model_name=args.model)
    # chat_lst_converter = LLMAPI.convert_chat_list
    # local_path = "Qwen/Qwen2.5-3B-Instruct"
    # from verl.utils import hf_tokenizer
    # tokenizer = hf_tokenizer(local_path)

    model = OpenAI(api_key=environment.OPENAI_API_KEY)

    results = []
    results_simplified = []

    for example in tqdm(data):

        response = model.chat.completions.create(
            model=os.path.basename(args.model),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example['reasonings'][0]}
            ]
        )
        response_text = response.choices[0].message.content

        analysis = json.loads(
                            re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL).group(1) 
                            if re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL) 
                            else response_text)
        # try:
        #     analysis['final_answer'] = int(re.findall(r'\d+', example['responses'][0])[0])
        # except:
        #     analysis['final_answer'] = 'Unparsed response: {}'.format(example['responses'])

        example['analysis'] = analysis
        example['ground_truths']['training_features'], example['ground_truths']['training_labels'], example['ground_truths']['true_centroids'] = compute_centroids(example)

        results.append(example)
        results_simplified.append({
            'analysis': analysis,
            'ground_truths': example['ground_truths']
        })
    
        with open('deepseek-ai-deepseek-reasoner/reasoning_analysis_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        with open('deepseek-ai-deepseek-reasoner/reasoning_analysis_results_simplified.json', 'w') as f:
            json.dump(results_simplified, f, indent=4)

        time.sleep(1) 