#!/usr/bin/env python3
import json
from classify_fitted_model import classify_model_function, read_prompt

def main():
    # Sample model code from the JSON file
    model_code = """def model(x, y, data):
    if x > 0 and y > 0:
        return 1
    return 0"""
    
    # Path to the classification prompt
    prompt_path = "/home/szhang967/liftr/reasoning_analysis/classifier_prompt_classfication.txt"
    
    # Read the prompt
    prompt = read_prompt(prompt_path)
    
    # Classify the model function
    classification = classify_model_function(model_code, prompt, llm_type="claude")
    
    # Print the classification result
    print("Model code:")
    print(model_code)
    print("\nClassification:")
    print(json.dumps(classification, indent=2))

if __name__ == "__main__":
    main() 