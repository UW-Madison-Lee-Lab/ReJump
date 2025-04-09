#!/usr/bin/env python3
"""
脚本用于生成ICL推理文本示例文件
在不同的任务类型、噪声水平和翻转率上生成一个或多个示例
"""

import os
import argparse
import random
import numpy as np
from pathlib import Path

# 导入数据生成函数
from examples.data_preprocess.moons import gen_dataset as gen_moons_dataset
from examples.data_preprocess.circles import gen_dataset as gen_circles_dataset
from examples.data_preprocess.blobs import gen_dataset as gen_blobs_dataset
from examples.data_preprocess.linear import gen_dataset as gen_linear_dataset


def get_task_description(task_type):
    """返回任务描述"""
    return {
        "moons": "月牙形分类问题",
        "circles": "同心圆分类问题",
        "blobs": "多类簇分类问题",
        "linear": "线性分类问题"
    }.get(task_type, "未知分类问题")


def get_num_classes(task_type):
    """返回任务类型的类别数"""
    return {
        "blobs": 3,
        "circles": 2,
        "linear": 2,
        "moons": 2
    }.get(task_type, 2)


def generate_data(task_type, num_samples, feature_noise, flip_rate, seed_value):
    """生成数据样本"""
    if task_type == "moons":
        samples = gen_moons_dataset(
            num_samples=num_samples,
            feature_noise=feature_noise,
            label_noise=flip_rate,
            seed_value=seed_value
        )
    elif task_type == "circles":
        samples = gen_circles_dataset(
            num_samples=num_samples,
            feature_noise=feature_noise,
            label_noise=flip_rate,
            seed_value=seed_value
        )
    elif task_type == "blobs":
        samples = gen_blobs_dataset(
            num_samples=num_samples,
            feature_noise=feature_noise,
            label_noise=flip_rate,
            seed_value=seed_value,
        )
    elif task_type == "linear":
        samples = gen_linear_dataset(
            num_samples=num_samples,
            feature_noise=feature_noise,
            label_noise=flip_rate,
            seed_value=seed_value
        )
    else:
        raise ValueError(f"未知的任务类型: {task_type}")
    
    return samples


def generate_reasoning(task_type, test_point, examples, num_classes):
    """生成推理过程的示例文本"""
    test_x, test_y = test_point
    
    if task_type == "moons":
        reasoning = f"""<think>
我需要分析月牙形数据集的模式。让我先观察提供的例子：

"""
        # 按类别整理样本
        examples_by_class = {i: [] for i in range(num_classes)}
        for features, label in examples:
            examples_by_class[label].append(features)
        
        # 添加类别观察
        for label in range(num_classes):
            reasoning += f"类别 {label} 的点:\n"
            for features in examples_by_class[label][:3]:  # 最多显示3个例子
                reasoning += f"- ({features[0]:.3f}, {features[1]:.3f})\n"
            reasoning += "\n"
        
        reasoning += f"""
观察这些点，我注意到月牙形的数据分布。类别0和类别1形成了两个相对的月牙形。

对于待分类的点 ({test_x[0]:.3f}, {test_x[1]:.3f})，我认为它属于类别{test_y}，因为它的位置更接近类别{test_y}的分布区域。
</think>"""
    
    elif task_type == "circles":
        reasoning = f"""<think>
我需要分析同心圆数据集的模式。让我观察提供的例子：

"""
        # 按类别整理样本
        examples_by_class = {i: [] for i in range(num_classes)}
        for features, label in examples:
            examples_by_class[label].append(features)
        
        # 添加类别观察
        for label in range(num_classes):
            reasoning += f"类别 {label} 的点:\n"
            for features in examples_by_class[label][:3]:  # 最多显示3个例子
                reasoning += f"- ({features[0]:.3f}, {features[1]:.3f})\n"
            reasoning += "\n"
        
        # 计算测试点到原点的距离
        distance = np.sqrt(test_x[0]**2 + test_x[1]**2)
        
        reasoning += f"""
观察这些点，我注意到这是一个同心圆分布。类别似乎是根据点到原点(0,0)的距离来确定的。

对于待分类的点 ({test_x[0]:.3f}, {test_x[1]:.3f})，计算其到原点的距离：
距离 = √({test_x[0]:.3f}² + {test_x[1]:.3f}²) = {distance:.3f}

基于这个距离和观察到的模式，我认为这个点属于类别{test_y}。
</think>"""
    
    elif task_type == "blobs":
        reasoning = f"""<think>
我需要分析多类簇数据集的模式。让我观察提供的例子：

"""
        # 按类别整理样本
        examples_by_class = {i: [] for i in range(num_classes)}
        for features, label in examples:
            examples_by_class[label].append(features)
        
        # 添加类别观察
        for label in range(num_classes):
            reasoning += f"类别 {label} 的点:\n"
            for features in examples_by_class[label][:3]:  # 最多显示3个例子
                reasoning += f"- ({features[0]:.3f}, {features[1]:.3f})\n"
            
            # 计算簇中心
            if examples_by_class[label]:
                center_x = sum(f[0] for f in examples_by_class[label]) / len(examples_by_class[label])
                center_y = sum(f[1] for f in examples_by_class[label]) / len(examples_by_class[label])
                reasoning += f"簇中心大约在: ({center_x:.3f}, {center_y:.3f})\n\n"
        
        reasoning += f"""
观察这些点，我注意到{num_classes}个不同的簇。

对于待分类的点 ({test_x[0]:.3f}, {test_x[1]:.3f})，根据它与各个簇中心的距离，我认为它属于类别{test_y}。
</think>"""
    
    elif task_type == "linear":
        reasoning = f"""<think>
我需要分析线性分类数据集的模式。让我观察提供的例子：

"""
        # 按类别整理样本
        examples_by_class = {i: [] for i in range(num_classes)}
        for features, label in examples:
            examples_by_class[label].append(features)
        
        # 添加类别观察
        for label in range(num_classes):
            reasoning += f"类别 {label} 的点:\n"
            for features in examples_by_class[label][:3]:  # 最多显示3个例子
                reasoning += f"- ({features[0]:.3f}, {features[1]:.3f})\n"
            reasoning += "\n"
        
        reasoning += f"""
观察这些点，我注意到这是一个线性分类问题。看起来存在一个线性决策边界将两类数据分开。

对于待分类的点 ({test_x[0]:.3f}, {test_x[1]:.3f})，根据它相对于决策边界的位置，我认为它属于类别{test_y}。
</think>"""
    
    else:
        reasoning = f"""<think>
我需要分析数据集的模式。让我观察提供的例子，找出特征和标签之间的关系。

待分类的点是 ({test_x[0]:.3f}, {test_x[1]:.3f})，基于我观察到的模式，我认为它属于类别{test_y}。
</think>"""
    
    return reasoning


def generate_example(task_type, num_context, num_shot, feature_noise, flip_rate, seed):
    """生成单个示例文本"""
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    
    # 获取类别数
    num_classes = get_num_classes(task_type)
    
    # 生成上下文样本
    context_samples = generate_data(
        task_type=task_type,
        num_samples=num_context,
        feature_noise=feature_noise,
        flip_rate=flip_rate,
        seed_value=seed
    )
    
    # 生成测试样本
    test_samples = generate_data(
        task_type=task_type,
        num_samples=num_shot + 1,  # 多生成一个作为测试点
        feature_noise=feature_noise,
        flip_rate=flip_rate,
        seed_value=seed + 100  # 不同的种子
    )
    
    # 分离示例样本和测试点
    shot_examples = test_samples[:-1]
    test_point = test_samples[-1]
    
    # 构建示例文本
    example_text = f"Problem: This is a classification task. You will be provided with examples of how a skilled reasoner classifies data points based on their features. Study the examples carefully to understand the reasoning process. Then, classify the new data point following a similar reasoning approach.\n\n"
    
    # 添加示例数据点
    example_text += f"The dataset has {num_classes} classes: {list(range(num_classes))}. We first provide you with some examples of how to classify data points.\n"
    for features, label in shot_examples:
        example_text += f"Features: {features[0]:.3f}, {features[1]:.3f}, Label: {label}\n"
    
    # 添加测试问题
    example_text += f"\nGiven the data point with features {test_point[0][0]:.3f}, {test_point[0][1]:.3f}, classify it into one of the possible classes.\n\n"
    
    # 添加推理过程
    reasoning = generate_reasoning(task_type, test_point, context_samples, num_classes)
    example_text += f"Reasoning: {reasoning}\n\n"
    
    # 添加最终答案
    example_text += f"<answer>{test_point[1]}</answer>"
    
    return example_text


def main():
    parser = argparse.ArgumentParser(description='生成ICL推理文本示例文件')
    parser.add_argument('--output', type=str, required=True, help='输出文件路径')
    parser.add_argument('--num_examples', type=int, default=1, help='生成的示例数量')
    parser.add_argument('--task_type', type=str, default='moons', choices=['moons', 'circles', 'blobs', 'linear'], help='任务类型')
    parser.add_argument('--num_context', type=int, default=20, help='上下文样本数量')
    parser.add_argument('--num_shot', type=int, default=5, help='示例样本数量')
    parser.add_argument('--feature_noise', type=float, default=0.1, help='噪声水平')
    parser.add_argument('--flip_rate', type=float, default=0.0, help='标签翻转率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    
    # 创建输出目录
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    examples = []
    for i in range(args.num_examples):
        # 每个示例使用不同的种子
        example_seed = args.seed + i
        example_text = generate_example(
            task_type=args.task_type,
            num_context=args.num_context,
            num_shot=args.num_shot,
            feature_noise=args.feature_noise,
            flip_rate=args.flip_rate,
            seed=example_seed
        )
        examples.append(f"Example {i+1}:\n{example_text}\n\n")
    
    # 写入文件
    with open(output_path, 'w') as f:
        f.write("\n".join(examples))
    
    print(f"已生成 {args.num_examples} 个示例，保存到 {args.output}")


if __name__ == "__main__":
    main() 