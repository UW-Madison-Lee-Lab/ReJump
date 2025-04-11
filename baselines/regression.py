import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from constants import supported_datasets
from environment import root_dir
import argparse
regressors = {
    "linearRegression": LinearRegression,
    "svr": SVR,
    "randomForest": RandomForestRegressor,
    "gradientBoosting": GradientBoostingRegressor,
}

optimal_regressors = {
    "linreg": lambda x: np.dot(x, np.ones(2)/2) + 0.0,
    "quadreg": lambda x: np.sum(x**2, axis=0) - 1.0,
    "cosreg": lambda x: np.sum(np.cos(2 * np.pi * x), axis=0) / 2 + 0.0,
    "expreg": lambda x: np.sum(np.exp(x * np.log(2)), axis=0) / 2 - 1.0,
    "l1normreg": lambda x: np.sum(2*np.abs(x), axis=0) / 2 - 1.0,
    "pwreg": lambda x: np.sum(np.piecewise(x, [x < -0.5, (x >= -0.5) & (x < 0.5), x >= 0.5], [lambda x: x - 0.5, 0, lambda x: x + 0.5])) / 2 
}

def get_dataset(df, i):
    samples_train = df["reward_model"].iloc[i]["ground_truth"]["in_context_samples"]
    X_train = np.array([sample["features"] for sample in samples_train])
    y_train = np.array([sample["target"] for sample in samples_train])
    
    X_test = df["reward_model"].iloc[i]["ground_truth"]["features"]
    y_test = df["reward_model"].iloc[i]["ground_truth"]["label"]
    return X_train, y_train, X_test, y_test

def get_performance(dataset_name): 
    df = pd.read_parquet(f"{root_dir}/datasets/{dataset_name}/50_shot/qwen-instruct/500_samples_{supported_datasets[dataset_name]['feature_noise']}_noise_0.0_flip_rate_default_mode/test_default.parquet")
    
    results = {}
    for i in range(len(df)):
        X_train, y_train, X_test, y_test = get_dataset(df, i)

        # Initialize dictionaries to store results
        r2_scores = {}
        mse_scores = {}
        
        # Train and evaluate each model
        for model_name in regressors:
            
            if not model_name in results:
                results[model_name] = {"y_true": [], "y_pred": []}
            
            model = regressors[model_name]()
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test.reshape(1, -1))
            
            results[model_name]["y_true"].append(y_test)
            results[model_name]["y_pred"].append(y_pred)
            
        if not "optimal" in results:
            results["optimal"] = {"y_true": [], "y_pred": []}

        results["optimal"]["y_true"].append(y_test)
        results["optimal"]["y_pred"].append(optimal_regressors[dataset_name](X_test))

    for model_name in results:
        r2_scores[model_name] = r2_score(np.array(results[model_name]["y_true"]), np.array(results[model_name]["y_pred"]))
        mse_scores[model_name] = mean_squared_error(np.array(results[model_name]["y_true"]), np.array(results[model_name]["y_pred"]))

    # Create DataFrames for results
    r2_df = pd.DataFrame({
        'Model': list(r2_scores.keys()),
        'R² Score': [f"{score:.3f}" for score in r2_scores.values()]
    })
    
    mse_df = pd.DataFrame({
        'Model': list(mse_scores.keys()),
        'MSE': [f"{score:.3f}" for score in mse_scores.values()]
    })
    
    print("R2 Score")
    print(r2_df)
    print("MSE")
    print(mse_df)
    return r2_df.T, mse_df.T

if __name__ == "__main__":
    datasets = list(supported_datasets.keys())
    regression_datasets = []
    for dataset in datasets:
        if supported_datasets[dataset]["type"] == "regression":
            regression_datasets.append(dataset)
            
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="pwreg", choices=regression_datasets)
    args = parser.parse_args()
    r2_df, mse_df = get_performance(args.dataset)
