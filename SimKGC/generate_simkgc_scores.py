import torch
import os
import argparse
from tqdm import tqdm
from collections import defaultdict

from predict import BertPredictor
from doc import Example
from dict_hub import build_dict
from logger_config import logger

def generate_scores(dataset_name, ckt_path):
    """
    Generates and saves SimKGC scores for the test set of a given dataset.

    Args:
        dataset_name (str): The name of the dataset (e.g., 'FB15k237' or 'WN18RR').
        ckt_path (str): Path to the SimKGC model checkpoint for that dataset.
    """
    data_dir = os.path.join('data', dataset_name)
    output_dir = os.path.join('predictions', dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'simkgc_scores.pt')

    # 1. Initialize BertPredictor
    predictor = BertPredictor()
    predictor.load(ckt_path, use_data_parallel=False)

    # 2. Load entity dictionary
    entity_dict = build_dict(os.path.join(data_dir, 'entities.txt'))
    all_entities = [Example(head_id='', relation='', tail_id=entity) for entity in entity_dict.keys()]

    # 3. Generate embeddings for all entities
    logger.info(f"[{dataset_name}] Generating embeddings for all entities...")
    all_entity_embeddings = predictor.predict_by_entities(all_entities)
    logger.info(f"[{dataset_name}] Generated {all_entity_embeddings.shape[0]} entity embeddings.")

    # 4. Load test set
    test_file = os.path.join(data_dir, 'test.txt')
    test_examples = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            h, r, t = line.strip().split('\t')
            test_examples.append(Example(head_id=h, relation=r, tail_id=t))

    # 5. Generate and save scores for each test query
    scores_dict = {}
    logger.info(f"[{dataset_name}] Generating scores for test queries...")
    for example in tqdm(test_examples, desc=f"Processing {dataset_name}"):
        hr_embedding, _ = predictor.predict_by_examples([example])
        scores = torch.matmul(hr_embedding, all_entity_embeddings.T).squeeze(0)
        query_key = (example.head_id, example.relation)
        scores_dict[query_key] = scores.cpu()

    # 6. Save the scores dictionary
    logger.info(f"[{dataset_name}] Saving scores to {output_path}...")
    torch.save(scores_dict, output_path)
    logger.info(f"[{dataset_name}] Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate SimKGC scores for a given dataset.')
    parser.add_argument('--dataset', type=str, required=True, choices=['FB15k237', 'WN18RR'], help='The name of the dataset.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the trained SimKGC model checkpoint.')
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint path not found at '{args.checkpoint}'.")
    else:
        generate_scores(args.dataset, args.checkpoint)