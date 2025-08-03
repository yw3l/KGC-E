#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import time
import os
import json

from torch.utils.data import DataLoader
from dataloader import TestDataset
from collections import defaultdict

from ogb.linkproppred import Evaluator

# Helper function to load dictionaries
def load_dict(file_path):
    """Loads a dictionary from a file with format: id\tvalue"""
    id_to_value = {}
    value_to_id = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            value = line.strip()
            id_to_value[i] = value
            value_to_id[value] = i
    return id_to_value, value_to_id

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, evaluator,
                 double_entity_embedding=False, 
                 double_relation_embedding=False, triple_relation_embedding=False, quad_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim

        if double_relation_embedding:
            self.relation_dim = hidden_dim*2
        elif triple_relation_embedding:
            self.relation_dim = hidden_dim*3
        elif quad_relation_embedding:
            self.relation_dim = hidden_dim*4
        else:
            self.relation_dim = hidden_dim

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'PairRE', 'RotatEv2', 'CompoundE']:
            raise ValueError('model %s not supported' % model_name)

        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')

        if model_name == 'PairRE' and (not double_relation_embedding):
            raise ValueError('PairRE should use --double_relation_embedding')

        if model_name == 'CompoundE' and (not triple_relation_embedding):
            raise ValueError('CompoundE should use --triple_relation_embedding')

        self.evaluator = evaluator

    def forward(self, sample, mode='single'):
        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1
            head = torch.index_select(self.entity_embedding, dim=0, index=sample[:,0]).unsqueeze(1)
            relation = torch.index_select(self.relation_embedding, dim=0, index=sample[:,1]).unsqueeze(1)
            tail = torch.index_select(self.entity_embedding, dim=0, index=sample[:,2]).unsqueeze(1)
        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            head = torch.index_select(self.entity_embedding, dim=0, index=head_part.view(-1)).view(batch_size, negative_sample_size, -1)
            relation = torch.index_select(self.relation_embedding, dim=0, index=tail_part[:, 1]).unsqueeze(1)
            tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part[:, 2]).unsqueeze(1)
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            head = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
            relation = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 1]).unsqueeze(1)
            tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size, negative_sample_size, -1)
        else:
            raise ValueError('mode %s not supported' % mode)

        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'PairRE': self.PairRE,
            'RotatEv2': self.RotatEv2,
            'CompoundE': self.CompoundE
        }
        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        return score

    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail
        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)
        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail
        score = score.sum(dim = 2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)
        phase_relation = relation/(self.embedding_range.item()/pi)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)
        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail
        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)
        score = self.gamma.item() - score.sum(dim = 2)
        return score

    def RotatEv2(self, head, relation, tail, mode, r_norm=None):
        pi = 3.14159265358979323846
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)
        phase_relation = relation/(self.embedding_range.item()/pi)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)
        re_relation_head, re_relation_tail = torch.chunk(re_relation, 2, dim=2)
        im_relation_head, im_relation_tail = torch.chunk(im_relation, 2, dim=2)
        re_score_head = re_head * re_relation_head - im_head * im_relation_head
        im_score_head = re_head * im_relation_head + im_head * re_relation_head
        re_score_tail = re_tail * re_relation_tail - im_tail * im_relation_tail
        im_score_tail = re_tail * im_relation_tail + im_tail * re_relation_tail
        re_score = re_score_head - re_score_tail
        im_score = im_score_head - im_score_tail
        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)
        score = self.gamma.item() - score.sum(dim = 2)
        return score

    def PairRE(self, head, relation, tail, mode):
        re_head, re_tail = torch.chunk(relation, 2, dim=2)
        head = F.normalize(head, 2, -1)
        tail = F.normalize(tail, 2, -1)
        score = head * re_head - tail * re_tail
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def CompoundE(self, head, relation, tail, mode):
        tail_scale, tail_translate, theta = torch.chunk(relation, 3, dim=2)
        theta, _ = torch.chunk(theta, 2, dim=2)
        head = F.normalize(head, 2, -1)
        tail = F.normalize(tail, 2, -1)
        pi = 3.14159265358979323846
        theta = theta/(self.embedding_range.item()/pi)
        re_rotation = torch.cos(theta)
        im_rotation = torch.sin(theta)
        re_rotation = re_rotation.unsqueeze(-1)
        im_rotation = im_rotation.unsqueeze(-1)
        tail = tail.view((tail.shape[0], tail.shape[1], -1, 2))
        tail_r = torch.cat((re_rotation * tail[:, :, :, 0:1], im_rotation * tail[:, :, :, 0:1]), dim=-1)
        tail_r += torch.cat((-im_rotation * tail[:, :, :, 1:], re_rotation * tail[:, :, :, 1:]), dim=-1)
        tail_r = tail_r.view((tail_r.shape[0], tail_r.shape[1], -1))
        tail_r += tail_translate
        tail_r *= tail_scale
        score = head - tail_r
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        model.train()
        optimizer.zero_grad()
        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()
        negative_score = model((positive_sample, negative_sample), mode=mode)
        if args.negative_adversarial_sampling:
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)
        positive_score = model(positive_sample)
        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)
        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()
        loss = (positive_sample_loss + negative_sample_loss)/2
        if args.regularization != 0.0:
            regularization = args.regularization * (
                model.entity_embedding.norm(p = 3)**3 + 
                model.relation_embedding.norm(p = 3).norm(p = 3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
        loss.backward()
        optimizer.step()
        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }
        return log

    @staticmethod
    def test_step(model, test_triples, args, high_frequency_relations, low_frequency_relations, random_sampling=False):
        model.eval()
        
        # --- RTAME Integration Setup ---
        alpha = 0.7  # Weight for CompoundE on high-frequency relations
        beta = 0.3   # Weight for CompoundE on low-frequency relations
        
        # Load SimKGC pre-computed scores and dictionaries based on the dataset
        simkgc_data_dir = f'../SimKGC/data/{args.dataset}'
        simkgc_scores_path = f'../SimKGC/predictions/{args.dataset}/simkgc_scores.pt'
        
        logging.info(f"Loading SimKGC scores from {simkgc_scores_path}...")
        if not os.path.exists(simkgc_scores_path):
            raise FileNotFoundError(f"SimKGC scores not found at {simkgc_scores_path}. Please run generate_simkgc_scores.py for dataset {args.dataset} first.")
        simkgc_scores = torch.load(simkgc_scores_path)
        
        logging.info(f"Loading entity and relation dictionaries for {args.dataset}...")
        id_to_entity, _ = load_dict(os.path.join(simkgc_data_dir, 'entities.txt'))
        id_to_relation, _ = load_dict(os.path.join(simkgc_data_dir, 'relations.txt'))
        # --- End of RTAME Integration Setup ---

        test_dataloader_head = DataLoader(
            TestDataset(test_triples, args, 'head-batch', random_sampling),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TestDataset.collate_fn
        )
        test_dataloader_tail = DataLoader(
            TestDataset(test_triples, args, 'tail-batch', random_sampling),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TestDataset.collate_fn
        )
        test_dataset_list = [test_dataloader_head, test_dataloader_tail]
        test_logs = defaultdict(list)
        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, mode in test_dataset:
                    if args.cuda:
                        positive_sample = positive_sample.cuda()
                        negative_sample = negative_sample.cuda()

                    batch_size = positive_sample.size(0)
                    score_compounde = model((positive_sample, negative_sample), mode)

                    # --- RTAME Ensembling Logic ---
                    # 1. Get query key (h_str, r_str)
                    head_id = positive_sample[0, 0].item()
                    relation_id = positive_sample[0, 1].item()
                    head_str = id_to_entity.get(head_id)
                    relation_str = id_to_relation.get(relation_id)
                    query_key = (head_str, relation_str)

                    # 2. Get SimKGC scores for the current batch
                    if query_key in simkgc_scores:
                        # Get pre-computed scores for all entities
                        scores_for_all_entities = simkgc_scores[query_key]
                        
                        # Gather scores for the specific positive and negative tails in this batch
                        if mode == 'tail-batch':
                            pos_and_neg_ids = torch.cat([positive_sample[:, 2].unsqueeze(1), negative_sample], dim=1).flatten()
                        else: # head-batch
                            pos_and_neg_ids = torch.cat([positive_sample[:, 0].unsqueeze(1), negative_sample], dim=1).flatten()

                        score_simkgc_flat = scores_for_all_entities[pos_and_neg_ids]
                        score_simkgc = score_simkgc_flat.view(batch_size, -1)
                        if args.cuda:
                            score_simkgc = score_simkgc.cuda()
                    else:
                        logging.warning(f"Query key {query_key} not found in SimKGC scores. Using zero scores as fallback.")
                        score_simkgc = torch.zeros_like(score_compounde)

                    # 3. Ensemble based on relation type
                    if relation_str in high_frequency_relations:
                        ensembled_score = alpha * score_compounde + (1 - alpha) * score_simkgc
                    else:
                        ensembled_score = beta * score_compounde + (1 - beta) * score_simkgc
                    # --- End of RTAME Ensembling Logic ---

                    if model.evaluator is not None: # OGB Evaluator
                        batch_results = model.evaluator.eval({'y_pred_pos': ensembled_score[:, 0], 'y_pred_neg': ensembled_score[:, 1:]})
                    else: # Manual evaluation for legacy datasets
                        # Sort scores and get ranks
                        sorted_scores, sorted_indices = torch.sort(ensembled_score, dim=1, descending=True)
                        # The rank of the positive sample is where its original index (0) appears in the sorted list
                        ranks = (sorted_indices == 0).nonzero(as_tuple=True)[1] + 1
                        
                        batch_results = {}
                        batch_results['mrr'] = (1.0 / ranks).cpu()
                        for k in [1, 3, 10]:
                            batch_results[f'hits@{k}'] = (ranks <= k).float().cpu()

                    for metric in batch_results:
                        test_logs[metric].append(batch_results[metric])

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))
                    step += 1

            metrics = {}
            for metric in test_logs:
                metrics[metric] = torch.cat(test_logs[metric]).mean().item()

        return metrics