from environment import WANDB_INFO

import os
import argparse
import wandb
import re
from typing import List, Optional
from constants import get_result_dir
import pdb

def download_artifacts(run_id: str, output_dir: Optional[str] = None) -> List[str]:
    """
    Download all artifacts associated with a specific W&B run.
    
    Args:
        run_id: The W&B run ID to download artifacts from
        output_dir: Optional directory to save artifacts to. If None, uses current directory.
        
    Returns:
        List of paths to downloaded artifacts
    """
    # Set up wandb API using credentials from environment
    api = wandb.Api()
    
    # Parse the run_id to get entity, project, and run name
    if "/" in run_id:
        parts = run_id.split("/")
        if len(parts) == 3:
            entity, project, run_name = parts
        else:
            raise ValueError(f"Invalid run_id format: {run_id}. Expected format: entity/project/run_name")
    else:
        # If only run_name is provided, use default entity and project from WANDB_INFO
        entity = WANDB_INFO.get("entity")
        project = WANDB_INFO.get("project")
        run_name = run_id
        
        if not entity or not project:
            raise ValueError("When providing only run_name, WANDB_INFO must contain 'entity' and 'project'")
    
    
    print(f"Downloading artifacts from run: {entity}/{project}/{run_name}")
    
    # Get the run
    run = api.run(f"{entity}/{project}/{run_name}")
    
    # Get all artifacts used or logged by this run
    artifacts = run.logged_artifacts()
    
    if not artifacts:
        print("No artifacts found for this run.")
        return []
    
    # Set default output directory if not provided
    if output_dir is None:
        configs = run.config
        output_dir = get_result_dir(
            dataset_name=configs["dataset_name"],
            model_name=configs["model_name"],
            shot=configs["shot"],
            template_type=configs["template_type"],
            response_length=configs["response_length"],
            num_samples=configs["num_samples"],
            feature_noise=configs["feature_noise"],
            label_noise=configs["label_noise"],
            train_step=configs["train_step"],
            data_mode=configs["data_mode"],
        )
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    downloaded_paths = []
    
    # Download each artifact
    for artifact in artifacts:
        print(f"Downloading artifact: {artifact.name} (type: {artifact.type})")
        
        # Download the artifact
        artifact_dir = artifact.download(root=output_dir)
        downloaded_paths.append(artifact_dir)
        
        print(f"Downloaded to: {artifact_dir}")
    
    print(f"Successfully downloaded {len(downloaded_paths)} artifacts")
    return downloaded_paths
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download artifacts from a W&B run")
    parser.add_argument("run_id", help="W&B run ID in format 'entity/project/run_name' or just 'run_name'")
    parser.add_argument("--output-dir", "-o", help="Directory to save artifacts to")
    
    args = parser.parse_args()
    
    download_artifacts(args.run_id, args.output_dir)
