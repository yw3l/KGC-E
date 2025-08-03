#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random

import numpy as np
import torch

from torch.utils.data import DataLoader

from model import KGEModel

from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator

from ogb.linkproppred import LinkPropPredDataset, Evaluator
from collections import defaultdict
from tqdm import tqdm
import time
from tensorboardX import SummaryWriter
import os.path as osp


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')

    parser.add_argument('--dataset', type=str, default='FB15k237', help='dataset name, e.g., FB15k237, WN18RR, ogbl-wikikg2')
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')
    parser.add_argument('-tr', '--triple_relation_embedding', action='store_true')
    parser.add_argument('-qr', '--quad_relation_embedding', action='store_true')

    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true',
                        help='Otherwise use subsampling weighting like in word2vec')

    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)

    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')

    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')

    parser.add_argument('--print_on_screen', action='store_true', help='log on screen or not')
    parser.add_argument('--ntriples_eval_train', type=int, default=200000, help='number of training triples to evaluate eventually')
    parser.add_argument('--neg_size_eval_train', type=int, default=500, help='number of negative samples when evaluating training triples')
    parser.add_argument('--relation_type', type=str, default='all', help='1-1, 1-n, n-1, n-n')
    return parser.parse_args(args)

def override_config(args):
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    args.dataset = argparse_dict['dataset']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.triple_relation_embedding = argparse_dict['triple_relation_embedding']
    args.quad_relation_embedding = argparse_dict['quad_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']

def save_model(model, optimizer, save_variable_list, args):
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)
    torch.save(
        {
            **save_variable_list,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        },
        os.path.join(args.save_path, 'checkpoint')
    )
    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(os.path.join(args.save_path, 'entity_embedding'), entity_embedding)
    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(os.path.join(args.save_path, 'relation_embedding'), relation_embedding)


def read_triples(file_path):
    triples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            h, r, t = line.strip().split('\t')
            triples.append((int(h), int(r), int(t)))
    return triples

def load_legacy_dataset(dataset_name):
    """Loads FB15k-237 and WN18RR datasets with robust file checking."""
    logging.info(f"Loading legacy dataset: {dataset_name}")
    data_dir = f'../SimKGC/data/{dataset_name}'

    def get_count(name):
        """Smartly gets entity or relation count from either .json or .txt files."""
        json_path = os.path.join(data_dir, f'{name}.json')
        txt_path = os.path.join(data_dir, f'{name}2id.txt')

        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                return len(json.load(f))
        elif os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                # The first line of entity2id.txt/relation2id.txt is the count
                return int(f.readline().strip())
        else:
            raise FileNotFoundError(
                f"Could not find entity/relation count file. Neither {json_path} nor {txt_path} exists. "
                f"Please ensure data is in place or run SimKGC's preprocess.sh script."
            )

    nentity = get_count('entities') if dataset_name == 'FB15k237' else get_count('entity')
    nrelation = get_count('relations') if dataset_name == 'FB15k237' else get_count('relation')

    train_triples = read_triples(os.path.join(data_dir, 'train.txt'))
    valid_triples = read_triples(os.path.join(data_dir, 'valid.txt'))
    test_triples = read_triples(os.path.join(data_dir, 'test.txt'))

    def to_split_dict(triples_list):
        heads = np.array([h for h, r, t in triples_list], dtype=np.int64)
        rels = np.array([r for h, r, t in triples_list], dtype=np.int64)
        tails = np.array([t for h, r, t in triples_list], dtype=np.int64)
        return {'head': heads, 'relation': rels, 'tail': tails}

    split_dict = {
        'train': to_split_dict(train_triples),
        'valid': to_split_dict(valid_triples),
        'test': to_split_dict(test_triples)
    }

    return split_dict, nentity, nrelation


def set_logger(args):
    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    if args.print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

def log_metrics(mode, step, metrics, writer):
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))
        writer.add_scalar("_" .join([mode, metric]), metrics[metric], step)


def main(args):
    if (not args.do_train) and (not args.do_valid) and (not args.do_test) and (not args.evaluate_train):
        raise ValueError('one of train/val/test mode must be choosed.')

    if args.init_checkpoint:
        override_config(args)

    args.save_path = 'log/%s/%s/%s-%s/%s'%(args.dataset, args.model, args.hidden_dim, args.gamma, time.time()) if args.save_path == None else args.save_path
    writer = SummaryWriter(args.save_path)
    set_logger(args)

    # Load relation types based on dataset
    if args.dataset == 'FB15k237':
        relation_type_file = 'relation_types.json'
    elif args.dataset == 'WN18RR':
        relation_type_file = 'relation_types_wn18rr.json'
    else:
        relation_type_file = None

    if relation_type_file:
        relation_type_path = os.path.join(os.path.dirname(__file__), '..', relation_type_file)
        logging.info(f"Loading relation types from {relation_type_path}...")
        with open(relation_type_path, 'r') as f:
            relation_types = json.load(f)
        high_frequency_relations = relation_types['high_frequency_relations']
        low_frequency_relations = relation_types['low_frequency_relations']
        logging.info(f"Loaded {len(high_frequency_relations)} high-frequency relations for {args.dataset}.")
        logging.info(f"Loaded {len(low_frequency_relations)} low-frequency relations for {args.dataset}.")
    else:
        high_frequency_relations, low_frequency_relations = [], [] # Not used for OGB datasets

    # Load data
    if args.dataset in ['FB15k237', 'WN18RR']:
        split_dict, nentity, nrelation = load_legacy_dataset(args.dataset)
        evaluator = None # We will use our own evaluator logic for legacy datasets
    else:
        dataset = LinkPropPredDataset(name = args.dataset)
        split_dict = dataset.get_edge_split()
        nentity = dataset.graph['num_nodes']
        nrelation = int(max(dataset.graph['edge_reltype'])[0])+1
        evaluator = Evaluator(name = args.dataset)

    args.nentity = nentity
    args.nrelation = nrelation

    logging.info('Model: %s' % args.model)
    logging.info('Dataset: %s' % args.dataset)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)

    train_triples = split_dict['train']
    logging.info('#train: %d' % len(train_triples['head']))
    valid_triples = split_dict['valid']
    logging.info('#valid: %d' % len(valid_triples['head']))
    test_triples = split_dict['test']
    logging.info('#test: %d' % len(test_triples['head']))

    kge_model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding,
        triple_relation_embedding=args.triple_relation_embedding,
        quad_relation_embedding=args.quad_relation_embedding,
        evaluator=evaluator # This will be None for legacy datasets
    )

    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.cuda:
        kge_model = kge_model.cuda()

    if args.do_train:
        train_count, train_true_head, train_true_tail = defaultdict(lambda: 4), defaultdict(list), defaultdict(list)
        for i in tqdm(range(len(train_triples['head']))):
            head, relation, tail = train_triples['head'][i], train_triples['relation'][i], train_triples['tail'][i]
            train_true_head[(relation, tail)].append(head)
            train_true_tail[(head, relation)].append(tail)

        train_dataloader_head = DataLoader(
            TrainDataset(train_triples, nentity, nrelation,
                args.negative_sample_size, 'head-batch',
                train_count, train_true_head, train_true_tail),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )
        train_dataloader_tail = DataLoader(
            TrainDataset(train_triples, nentity, nrelation,
                args.negative_sample_size, 'tail-batch',
                train_count, train_true_head, train_true_tail),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )
        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()),
            lr=current_learning_rate
        )
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

    if args.init_checkpoint:
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0

    step = init_step
    if args.do_train:
        logging.info('Start Training...')
        logging.info('init_step = %d' % init_step)
        training_logs = []
        for step in range(init_step, args.max_steps):
            log = kge_model.train_step(kge_model, optimizer, train_iterator, args)
            training_logs.append(log)
            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()),
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3
            if step % args.save_checkpoint_steps == 0 and step > 0:
                save_variable_list = {
                    'step': step,
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(kge_model, optimizer, save_variable_list, args)
            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                log_metrics('Train', step, metrics, writer)
                training_logs = []
            if args.do_valid and step % args.valid_steps == 0 and step > 0:
                logging.info('Evaluating on Valid Dataset...')
                metrics = kge_model.test_step(kge_model, valid_triples, args, high_frequency_relations, low_frequency_relations)
                log_metrics('Valid', step, metrics, writer)
        save_variable_list = {
            'step': step,
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(kge_model, optimizer, save_variable_list, args)

    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics = kge_model.test_step(kge_model, valid_triples, args, high_frequency_relations, low_frequency_relations)
        log_metrics('Valid', step, metrics, writer)

    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        metrics = kge_model.test_step(kge_model, test_triples, args, high_frequency_relations, low_frequency_relations)
        log_metrics('Test', step, metrics, writer)

if __name__ == '__main__':
    main(parse_args())