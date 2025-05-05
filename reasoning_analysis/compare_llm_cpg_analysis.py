#!/usr/bin/env python3
"""
Compare LLM cognitive process graph analysis results across different models or settings.

This script reads multiple LLM analysis JSON files and computes similarity metrics between
their node type counts, average probabilities, and dependency transition matrices.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import itertools
from collections import OrderedDict
import pandas as pd
import os


def load_analysis_file(file_path: str) -> Dict[str, Any]:
    """
    Load an LLM analysis JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary containing the JSON data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_metadata_vectors(metadata: Dict[str, Any]) -> Tuple[OrderedDict, Dict[str, Any]]:
    """
    Extract node type counts, avg_prob and dependency matrices from metadata.
    Also extract node types in a consistent order.
    
    Args:
        metadata: Metadata dictionary from an LLM analysis file
        
    Returns:
        Tuple of (ordered_node_types, metrics_dict)
    """
    metrics = metadata.get("average_summary_metrics", {})
    
    # Get all node types and sort them alphabetically for consistent ordering
    node_types = set()
    for key in ["node_type_counts", "node_avg_prob_sum", "dependency"]:
        if key in metrics:
            node_types.update(metrics[key].keys())
    
    # Create an ordered dictionary of node types for consistent indexing
    ordered_node_types = OrderedDict([(node_type, i) for i, node_type in enumerate(sorted(node_types))])
    
    return ordered_node_types, metrics


def create_vectors_and_matrices(ordered_node_types: OrderedDict, metrics: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Create vectors and matrices for node_type_counts, node_avg_prob_sum, and dependency.
    
    Args:
        ordered_node_types: OrderedDict mapping node types to indices
        metrics: Dictionary containing the metrics
        
    Returns:
        Dictionary of numpy arrays for each metric
    """
    num_node_types = len(ordered_node_types)
    result = {}
    
    # Create node_type_counts vector
    if "node_type_counts" in metrics:
        counts_vector = np.zeros(num_node_types)
        for node_type, count in metrics["node_type_counts"].items():
            if node_type in ordered_node_types:
                counts_vector[ordered_node_types[node_type]] = count
        result["node_type_counts"] = counts_vector
    
    # Create node_avg_prob_sum vector
    if "node_avg_prob_sum" in metrics:
        prob_vector = np.zeros(num_node_types)
        for node_type, prob in metrics["node_avg_prob_sum"].items():
            if node_type in ordered_node_types and prob is not None:
                prob_vector[ordered_node_types[node_type]] = prob
        result["node_avg_prob_sum"] = prob_vector
    
    # Create dependency matrix
    if "dependency" in metrics:
        dep_matrix = np.zeros((num_node_types, num_node_types))
        for from_type, to_types in metrics["dependency"].items():
            if from_type in ordered_node_types:
                for to_type, value in to_types.items():
                    if to_type in ordered_node_types:
                        # Value could be a number or a dict with "num" key
                        if isinstance(value, dict) and "num" in value:
                            dep_matrix[ordered_node_types[from_type], ordered_node_types[to_type]] = value["num"]
                        else:
                            dep_matrix[ordered_node_types[from_type], ordered_node_types[to_type]] = value
        result["dependency"] = dep_matrix
    
    # Create confidence transition matrix
    # 首先检查node_confidence_transitions是否存在，并记录其结构
    if "all_confidence_transitions" in metrics or "node_confidence_transitions" in metrics:
        # 确定哪个键包含信息
#         import pdb; pdb.set_trace()
        transitions_key = "all_confidence_transitions"
        transitions = metrics[transitions_key]
        
        print(f"Found {len(transitions)} transitions in {transitions_key}")
        if transitions and len(transitions) > 0:
            # 显示第一个转换的结构
            print(f"First transition structure: {transitions[0]}")
        
        # Each cell will contain the average confidence value for that transition
        source_prob_matrix = np.zeros((num_node_types, num_node_types))
        target_prob_matrix = np.zeros((num_node_types, num_node_types))
        source_to_target_prob_difference_matrix = np.zeros((num_node_types, num_node_types))
        transition_counts = np.zeros((num_node_types, num_node_types))
        
        for transition in transitions:
            source_type = transition.get("source_node")
            target_type = transition.get("target_node")
            source_prob = transition.get("source_avg_prob")
            target_prob = transition.get("target_avg_prob")
            
            if (source_type in ordered_node_types and target_type in ordered_node_types and 
                source_prob is not None and target_prob is not None):
                source_idx = ordered_node_types[source_type]
                target_idx = ordered_node_types[target_type]
                
                source_prob_matrix[source_idx, target_idx] += source_prob
                target_prob_matrix[source_idx, target_idx] += target_prob
                source_to_target_prob_difference_matrix[source_idx, target_idx] += (source_prob - target_prob)
                transition_counts[source_idx, target_idx] += 1
        
        # Calculate averages only for cells with counts > 0
        nonzero_mask = transition_counts > 0
        if np.any(nonzero_mask):  # 确保至少有一个非零元素
            source_prob_matrix[nonzero_mask] /= transition_counts[nonzero_mask]
            target_prob_matrix[nonzero_mask] /= transition_counts[nonzero_mask]
            source_to_target_prob_difference_matrix[nonzero_mask] /= transition_counts[nonzero_mask]
            
            result["source_confidence"] = source_prob_matrix
            result["target_confidence"] = target_prob_matrix
            result["confidence_difference"] = source_to_target_prob_difference_matrix
            print(f"Source confidence matrix shape: {source_prob_matrix.shape}")
            print(f"Target confidence matrix shape: {target_prob_matrix.shape}")
            print(f"Confidence difference matrix shape: {source_to_target_prob_difference_matrix.shape}")
        else:
            print("Warning: No valid transitions found in the data")
    
    return result


def compute_cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        Cosine similarity in [0, 1]
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0
    
    return np.dot(v1, v2) / (norm1 * norm2)


def compute_matrix_similarity(m1: np.ndarray, m2: np.ndarray) -> float:
    """
    Compute similarity between two matrices using Frobenius inner product.
    
    Args:
        m1: First matrix
        m2: Second matrix
        
    Returns:
        Normalized similarity in [0, 1]
    """
    # Flatten matrices and compute cosine similarity
    v1 = m1.flatten()
    v2 = m2.flatten()
    return compute_cosine_similarity(v1, v2)


def save_detailed_analysis(file_paths: List[str], all_vectors_matrices: List[Dict[str, np.ndarray]], similarities: Dict[str, Dict[Tuple[str, str], float]], ordered_node_types: OrderedDict) -> str:
    """
    保存详细分析结果到文本文件
    
    Args:
        file_paths: 分析的文件路径列表
        all_vectors_matrices: 所有文件的向量和矩阵数据
        similarities: 相似度计算结果
        ordered_node_types: 有序的节点类型字典
        
    Returns:
        保存的文件路径
    """
    # 创建输出目录
    script_dir = Path(__file__).parent
    output_dir = script_dir / "cpg_detailed_analysis"
    output_dir.mkdir(exist_ok=True)
    
    # 创建输出文件
    output_file = output_dir / f"cpg_analysis.txt"
    
    # 打印调试信息
    print("\nAvailable data for analysis:")
    if all_vectors_matrices and len(all_vectors_matrices) > 0:
        first_file = all_vectors_matrices[0]
        print(f"Keys in first file: {list(first_file.keys())}")
    else:
        print("No vector matrices data available")
    
    with open(output_file, "w") as f:
        # 写入标题
        f.write("=" * 100 + "\n")
        f.write("LLM Cognitive Process Graph Analysis - Detailed Report\n")
        f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 100 + "\n\n")
        
        # 写入文件列表
        f.write("Analyzed Files:\n")
        f.write("-" * 80 + "\n")
        for i, filepath in enumerate(file_paths):
            f.write(f"{i+1}. {filepath}\n")
        f.write("\n\n")
        
        # 写入节点类型
        f.write("Node Types (in alphabetical order):\n")
        f.write("-" * 50 + "\n")
        for i, node_type in enumerate(ordered_node_types.keys()):
            f.write(f"{i+1}. {node_type}\n")
        f.write("\n\n")
        
        # 写入每个文件的数据
        f.write("Individual File Analysis:\n")
        f.write("=" * 100 + "\n\n")
        
        for i, (filepath, vectors_matrices) in enumerate(zip(file_paths, all_vectors_matrices)):
            f.write(f"File {i+1}: {filepath}\n")
            f.write("-" * 80 + "\n\n")
            
            # 添加可用的数据键列表
            f.write(f"Available data keys: {list(vectors_matrices.keys())}\n\n")
            
            # 节点类型计数向量
            if "node_type_counts" in vectors_matrices:
                f.write("Node Type Counts:\n")
                for node_type, idx in ordered_node_types.items():
                    count = vectors_matrices["node_type_counts"][idx]
                    f.write(f"  {node_type}: {count:.4f}\n")
                f.write("\n")
            
            # 节点平均概率
            if "node_avg_prob_sum" in vectors_matrices:
                f.write("Node Average Probabilities:\n")
                for node_type, idx in ordered_node_types.items():
                    prob = vectors_matrices["node_avg_prob_sum"][idx]
                    f.write(f"  {node_type}: {prob:.4f}\n")
                f.write("\n")
            
            # 依赖矩阵
            if "dependency" in vectors_matrices:
                f.write("Dependency Matrix:\n")
                f.write("  " + " ".join(f"{node_type[:10]:>10}" for node_type in ordered_node_types.keys()) + "\n")
                
                for from_type, from_idx in ordered_node_types.items():
                    row_values = [f"{vectors_matrices['dependency'][from_idx, to_idx]:.2f}" for to_idx in range(len(ordered_node_types))]
                    f.write(f"{from_type[:10]:>10} " + " ".join(f"{val:>10}" for val in row_values) + "\n")
                f.write("\n")
            
            # 源置信度矩阵
            if "source_confidence" in vectors_matrices:
                f.write("Source Confidence Matrix:\n")
                f.write("  " + " ".join(f"{node_type[:10]:>10}" for node_type in ordered_node_types.keys()) + "\n")
                
                for from_type, from_idx in ordered_node_types.items():
                    row_values = [f"{vectors_matrices['source_confidence'][from_idx, to_idx]:.4f}" for to_idx in range(len(ordered_node_types))]
                    f.write(f"{from_type[:10]:>10} " + " ".join(f"{val:>10}" for val in row_values) + "\n")
                f.write("\n")
            else:
                f.write("Source Confidence Matrix: Not available in this data\n\n")
            
            # 目标置信度矩阵
            if "target_confidence" in vectors_matrices:
                f.write("Target Confidence Matrix:\n")
                f.write("  " + " ".join(f"{node_type[:10]:>10}" for node_type in ordered_node_types.keys()) + "\n")
                
                for from_type, from_idx in ordered_node_types.items():
                    row_values = [f"{vectors_matrices['target_confidence'][from_idx, to_idx]:.4f}" for to_idx in range(len(ordered_node_types))]
                    f.write(f"{from_type[:10]:>10} " + " ".join(f"{val:>10}" for val in row_values) + "\n")
                f.write("\n")
            else:
                f.write("Target Confidence Matrix: Not available in this data\n\n")
            
            # 源到目标置信度差异矩阵
            if "confidence_difference" in vectors_matrices:
                f.write("Source-Target Confidence Difference Matrix (Source - Target):\n")
                f.write("  " + " ".join(f"{node_type[:10]:>10}" for node_type in ordered_node_types.keys()) + "\n")
                
                for from_type, from_idx in ordered_node_types.items():
                    row_values = [f"{vectors_matrices['confidence_difference'][from_idx, to_idx]:.4f}" for to_idx in range(len(ordered_node_types))]
                    f.write(f"{from_type[:10]:>10} " + " ".join(f"{val:>10}" for val in row_values) + "\n")
                f.write("\n")
            else:
                f.write("Confidence Difference Matrix: Not available in this data\n\n")
            
            f.write("\n" + "=" * 100 + "\n\n")
        
        # 写入相似度比较
        f.write("Similarity Analysis:\n")
        f.write("=" * 100 + "\n\n")
        
        for metric, pairs in similarities.items():
            if not pairs:
                continue
                
            f.write(f"{metric} Similarities:\n")
            f.write("-" * 80 + "\n")
            
            for (file1, file2), similarity in pairs.items():
                # 提取文件名部分，避免路径太长导致输出难以阅读
                file1_name = Path(file1).name
                file2_name = Path(file2).name
                f.write(f"  {file1_name} vs {file2_name}: {similarity:.4f}\n")
                f.write(f"    Full paths:\n")
                f.write(f"    - {file1}\n")
                f.write(f"    - {file2}\n\n")
            
            # 计算平均相似度
            avg_similarity = sum(pairs.values()) / len(pairs)
            f.write(f"\n  Average {metric} similarity: {avg_similarity:.4f}\n\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("End of Report\n")
    
    return str(output_file)


def extract_and_process_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts and processes metadata from analysis file, with special handling for transitions data
    
    Args:
        data: The loaded JSON data
        
    Returns:
        Processed metadata with additional metrics if needed
    """
    if "metadata" not in data:
        return {}
        
    metadata = data["metadata"]
    
    # Process confidence transitions data
    if "all_confidence_transitions" in metadata:
        # If we have transitions data at the metadata level rather than in average_summary_metrics
        if "average_summary_metrics" not in metadata:
            metadata["average_summary_metrics"] = {}
            
        # Add the transitions to average_summary_metrics for easier processing
        metadata["average_summary_metrics"]["all_confidence_transitions"] = metadata["all_confidence_transitions"]
        
        print(f"Found {len(metadata['all_confidence_transitions'])} transitions at metadata level")
        if len(metadata["all_confidence_transitions"]) > 0:
            print(f"Sample transition: {metadata['all_confidence_transitions'][0]}")
    
    return metadata


def compare_analysis_files(file_paths: List[str]):
    """
    Compare multiple LLM analysis files and calculate similarities.
    
    Args:
        file_paths: List of paths to LLM analysis JSON files
        
    Returns:
        Dictionary mapping metric names to dictionaries of file pairs to similarity scores
    """
    # Load all files
    analysis_data = []
    file_paths_valid = []
    
    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            print(f"Warning: File not found: {file_path}")
            continue
            
        try:
            data = load_analysis_file(file_path)
            # Process metadata to ensure transitions data is available
            processed_data = {"metadata": extract_and_process_metadata(data)}
            
            if not processed_data["metadata"]:
                print(f"Warning: No valid metadata found in file: {file_path}")
                continue
                
            file_paths_valid.append(str(path))
            analysis_data.append(processed_data)
            print(f"Successfully loaded and processed: {path.name}")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue
    
    if len(analysis_data) < 2:
        print("Error: At least two valid analysis files are required for comparison")
        return {}, [], [], OrderedDict()
    
    # Collect all node types across all files
    all_node_types = set()
    for data in analysis_data:
        metadata = data.get("metadata", {})
        metrics = metadata.get("average_summary_metrics", {})
        
        for key in ["node_type_counts", "node_avg_prob_sum", "dependency"]:
            if key in metrics:
                all_node_types.update(metrics[key].keys())
    
    print(f"Found {len(all_node_types)} node types across all files: {sorted(list(all_node_types))}")
    
    # Create an ordered dictionary of all node types
    all_ordered_node_types = OrderedDict([(node_type, i) for i, node_type in enumerate(sorted(all_node_types))])
    
    # Extract vectors and matrices for each file
    all_vectors_matrices = []
    for data in analysis_data:
        metadata = data.get("metadata", {})
        metrics = metadata.get("average_summary_metrics", {})
        metrics["all_confidence_transitions"] = metadata.get("all_confidence_transitions", [])
        vectors_matrices = create_vectors_and_matrices(all_ordered_node_types, metrics)
        all_vectors_matrices.append(vectors_matrices)
    
    # Calculate similarities between each pair of files
    similarities = {
        "node_type_counts": {},
        "node_avg_prob_sum": {},
        "dependency": {},
        "source_confidence": {},
        "target_confidence": {},
        "confidence_difference": {}
    }
    
    for i, j in itertools.combinations(range(len(file_paths_valid)), 2):
        file_pair = (file_paths_valid[i], file_paths_valid[j])
        vm_i = all_vectors_matrices[i]
        vm_j = all_vectors_matrices[j]
        
        # Compare node_type_counts
        if "node_type_counts" in vm_i and "node_type_counts" in vm_j:
            similarities["node_type_counts"][file_pair] = compute_cosine_similarity(
                vm_i["node_type_counts"], vm_j["node_type_counts"]
            )
        
        # Compare node_avg_prob_sum
        if "node_avg_prob_sum" in vm_i and "node_avg_prob_sum" in vm_j:
            similarities["node_avg_prob_sum"][file_pair] = compute_cosine_similarity(
                vm_i["node_avg_prob_sum"], vm_j["node_avg_prob_sum"]
            )
        
        # Compare dependency matrices
        if "dependency" in vm_i and "dependency" in vm_j:
            similarities["dependency"][file_pair] = compute_matrix_similarity(
                vm_i["dependency"], vm_j["dependency"]
            )
        
        # Compare source_confidence matrices
        if "source_confidence" in vm_i and "source_confidence" in vm_j:
            similarities["source_confidence"][file_pair] = compute_matrix_similarity(
                vm_i["source_confidence"], vm_j["source_confidence"]
            )
        
        # Compare target_confidence matrices
        if "target_confidence" in vm_i and "target_confidence" in vm_j:
            similarities["target_confidence"][file_pair] = compute_matrix_similarity(
                vm_i["target_confidence"], vm_j["target_confidence"]
            )
            
        # Compare confidence_difference matrices
        if "confidence_difference" in vm_i and "confidence_difference" in vm_j:
            similarities["confidence_difference"][file_pair] = compute_matrix_similarity(
                vm_i["confidence_difference"], vm_j["confidence_difference"]
            )
    
    return similarities, file_paths_valid, all_vectors_matrices, all_ordered_node_types


def get_default_files():
    """
    递归查找当前目录及其子目录下所有以_logical_graph_llm_analysis.json结尾的文件
    """
#     script_dir = Path(__file__).parent
    script_dir ="/staging/szhang967/results_temperature_0"
    result = []
    
    for root, dirs, files in os.walk(script_dir):
        for file in files:
            if file.endswith('cognitive_process_graph_logical_graph_llm_analysis.json'):
                result.append(os.path.join(root, file))
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Compare LLM analysis files')
    parser.add_argument('--files', type=str, nargs='+', required=False,
                        default=get_default_files(),
                        help='Paths to LLM analysis JSON files (default: all *_logical_graph_llm_analysis.json files in current directory)')
    parser.add_argument('--output-txt', action='store_true',
                        help='Generate detailed analysis text file')
    
    args = parser.parse_args()
    
    # Check if any files were provided
    if not args.files:
        print("Error: No input files provided")
        return
    
    # Compare the files
    similarities, file_paths_valid, all_vectors_matrices, all_ordered_node_types = compare_analysis_files(args.files)
    
    # Print the results
    print("\nSimilarity Analysis Results:")
    print("============================")
    
    for metric, pairs in similarities.items():
        print(f"\n{metric} Similarities:")
        print("-" * (len(metric) + 13))
        
        for (file1, file2), similarity in pairs.items():
            # 使用Path.name提取文件名，但保留完整路径用于详细输出
            file1_name = Path(file1).name
            file2_name = Path(file2).name
            print(f"{file1_name} vs {file2_name}: {similarity:.4f}")
    
    # Print summary
    print("\nSummary:")
    print("========")
    
    for metric in similarities.keys():
        if similarities[metric]:
            avg_similarity = sum(similarities[metric].values()) / len(similarities[metric])
            print(f"Average {metric} similarity: {avg_similarity:.4f}")
    
    # 保存详细分析到文本文件 (默认始终生成)
    if True:
        # 直接使用从compare_analysis_files返回的数据
        output_file = save_detailed_analysis(file_paths_valid, all_vectors_matrices, similarities, all_ordered_node_types)
        print(f"\nDetailed analysis saved to: {output_file}")


if __name__ == "__main__":
    main() 