import collections
import argparse
import os

def analyze_frequency(dataset_name):
    """
    Analyzes the frequency of relations in a specified knowledge graph dataset.

    Args:
        dataset_name (str): The name of the dataset (e.g., 'FB15k237' or 'WN18RR').
    """
    file_path = os.path.join('KGC-E', 'SimKGC', 'data', dataset_name, 'train.txt')
    
    if not os.path.exists(file_path):
        print(f"Error: Training file not found at {file_path}")
        return

    relation_counts = collections.Counter()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                relation = parts[1]
                relation_counts[relation] += 1

    sorted_relations = relation_counts.most_common()

    print(f"--- Relation Frequency Analysis for {dataset_name} ---")
    for relation, count in sorted_relations:
        print(f"{relation}\t{count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze relation frequency for a given dataset.')
    parser.add_argument('--dataset', type=str, required=True, choices=['FB15k237', 'WN18RR'], help='The name of the dataset to analyze.')
    args = parser.parse_args()
    
    analyze_frequency(args.dataset)