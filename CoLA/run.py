import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from CSTs import compute_custom_statistic, compute_custom_statistic_base, compute_custom_statistic_equal
from model import Model
from utils import *

from sklearn.metrics import roc_auc_score
import random
import os
import dgl

import argparse
from tqdm import tqdm
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def padding_args(subgraph_size = 4,Dataset='BlogCatalog'):

    # Set argument
    parser = argparse.ArgumentParser(description='CoLA: Self-Supervised Contrastive Learning for Anomaly Detection')
    parser.add_argument('--dataset', type=str, default=Dataset)  # 'BlogCatalog'  'Flickr'  'ACM'  'cora'  'citeseer'  'pubmed'
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--num_epoch', type=int,default=100)
    parser.add_argument('--drop_prob', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=300)
    parser.add_argument('--subgraph_size', type=int, default=subgraph_size)
    parser.add_argument('--readout', type=str, default='avg')  #max min avg  weighted_sum
    parser.add_argument('--auc_test_rounds', type=int, default=300)
    parser.add_argument('--negsamp_ratio', type=int, default=1)

    return parser

import time

# Start the timer
def train_test_model(args, output_file):


    if args.lr is None:
        if args.dataset in ['cora','citeseer','pubmed','Flickr']:
            args.lr = 1e-3
        elif args.dataset == 'ACM':
            args.lr = 5e-4
        elif args.dataset == 'BlogCatalog':
            args.lr = 3e-3

    if args.num_epoch is None:
        if args.dataset in ['cora','citeseer','pubmed']:
            args.num_epoch = 100
        elif args.dataset in ['BlogCatalog','Flickr','ACM']:
            args.num_epoch = 400

    batch_size = args.batch_size
    subgraph_size = args.subgraph_size

    
    print('Dataset: ', args.dataset, file=output_file)
    first_epoch_loss = 0
    last_epoch_loss = 0

    # Set random seed
    dgl.random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load and preprocess data
    adj, features, labels, idx_train, idx_val,\
    idx_test, ano_label, str_ano_label, attr_ano_label = load_mat(args.dataset)

    features, _ = preprocess_features(features)

    dgl_graph = adj_to_dgl_graph(adj)

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = labels.shape[1]

    adj = normalize_adj(adj)
    adj = (adj + sp.eye(adj.shape[0])).todense()

    features = torch.FloatTensor(features[np.newaxis])
    adj = torch.FloatTensor(adj[np.newaxis])
    labels = torch.FloatTensor(labels[np.newaxis])
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # Initialize model and optimiser
    model = Model(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if torch.cuda.is_available():
        print('Using CUDA')
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    if torch.cuda.is_available():
        b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).cuda())
    else:
        b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]))
    xent = nn.CrossEntropyLoss()
    cnt_wait = 0
    best = 1e9
    best_t = 0
    batch_num = nb_nodes // batch_size + 1

    added_adj_zero_row = torch.zeros((nb_nodes, 1, subgraph_size))
    added_adj_zero_col = torch.zeros((nb_nodes, subgraph_size + 1, 1))
    added_adj_zero_col[:,-1,:] = 1.
    added_feat_zero_row = torch.zeros((nb_nodes, 1, ft_size))
    if torch.cuda.is_available():
        added_adj_zero_row = added_adj_zero_row.cuda()
        added_adj_zero_col = added_adj_zero_col.cuda()
        added_feat_zero_row = added_feat_zero_row.cuda()

    # Train model
    with tqdm(total=args.num_epoch, disable=True) as pbar:
        pbar.set_description('Training')
        for epoch in range(args.num_epoch):

            loss_full_batch = torch.zeros((nb_nodes,1))
            if torch.cuda.is_available():
                loss_full_batch = loss_full_batch.cuda()

            model.train()

            all_idx = list(range(nb_nodes))
            random.shuffle(all_idx)
            total_loss = 0.

            subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)

            for batch_idx in range(batch_num):

                optimiser.zero_grad()

                is_final_batch = (batch_idx == (batch_num - 1))

                if not is_final_batch:
                    idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                else:
                    idx = all_idx[batch_idx * batch_size:]

                cur_batch_size = len(idx)

                lbl = torch.unsqueeze(torch.cat((torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio))), 1)

                ba = []
                bf = []
                added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size))
                added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1))
                added_adj_zero_col[:, -1, :] = 1.
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))

                if torch.cuda.is_available():
                    lbl = lbl.cuda()
                    added_adj_zero_row = added_adj_zero_row.cuda()
                    added_adj_zero_col = added_adj_zero_col.cuda()
                    added_feat_zero_row = added_feat_zero_row.cuda()

                for i in idx:
                    cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_feat = features[:, subgraphs[i], :]
                    ba.append(cur_adj)
                    bf.append(cur_feat)

                ba = torch.cat(ba)
                ba = torch.cat((ba, added_adj_zero_row), dim=1)
                ba = torch.cat((ba, added_adj_zero_col), dim=2)
                bf = torch.cat(bf)
                bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]),dim=1)

                logits = model(bf, ba)
                loss_all = b_xent(logits, lbl)

                loss = torch.mean(loss_all)

                loss.backward()
                optimiser.step()

                loss = loss.detach().cpu().numpy()
                loss_full_batch[idx] = loss_all[: cur_batch_size].detach()

                if not is_final_batch:
                    total_loss += loss

            mean_loss = (total_loss * batch_size + loss * cur_batch_size) / nb_nodes
            if epoch == 0:
                first_epoch_loss = mean_loss
            if epoch == args.num_epoch - 1:
                last_epoch_loss = mean_loss
            print('Epoch:{} Loss:{:.8f}'.format(epoch, mean_loss), flush=True)
            print('Epoch:{} Loss:{:.8f}'.format(epoch, mean_loss), file=output_file)

            if mean_loss < best:
                best = mean_loss
                best_t = epoch
                cnt_wait = 0
                torch.save(model.state_dict(), 'best_model.pkl')
            else:
                cnt_wait += 1

            pbar.set_postfix(loss=mean_loss)
            pbar.update(1)
    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('best_model.pkl'))

    multi_round_ano_score = np.zeros((args.auc_test_rounds, nb_nodes))
    multi_round_ano_score_p = np.zeros((args.auc_test_rounds, nb_nodes))
    multi_round_ano_score_n = np.zeros((args.auc_test_rounds, nb_nodes))

    with tqdm(total=args.auc_test_rounds) as pbar_test:
        pbar_test.set_description('Testing')
        for round in range(args.auc_test_rounds):

            all_idx = list(range(nb_nodes))
            random.shuffle(all_idx)

            subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)

            for batch_idx in range(batch_num):

                optimiser.zero_grad()

                is_final_batch = (batch_idx == (batch_num - 1))

                if not is_final_batch:
                    idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                else:
                    idx = all_idx[batch_idx * batch_size:]

                cur_batch_size = len(idx)

                ba = []
                bf = []
                added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size))
                added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1))
                added_adj_zero_col[:, -1, :] = 1.
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))

                if torch.cuda.is_available():
                    lbl = lbl.cuda()
                    added_adj_zero_row = added_adj_zero_row.cuda()
                    added_adj_zero_col = added_adj_zero_col.cuda()
                    added_feat_zero_row = added_feat_zero_row.cuda()

                for i in idx:
                    cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_feat = features[:, subgraphs[i], :]
                    ba.append(cur_adj)
                    bf.append(cur_feat)

                ba = torch.cat(ba)
                ba = torch.cat((ba, added_adj_zero_row), dim=1)
                ba = torch.cat((ba, added_adj_zero_col), dim=2)
                bf = torch.cat(bf)
                bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)

                with torch.no_grad():
                    logits = torch.squeeze(model(bf, ba))
                    logits = torch.sigmoid(logits)

                ano_score = - (logits[:cur_batch_size] - logits[cur_batch_size:]).cpu().numpy()
                # ano_score_p = - logits[:cur_batch_size].cpu().numpy()
                # ano_score_n = logits[cur_batch_size:].cpu().numpy()

                multi_round_ano_score[round, idx] = ano_score
                # multi_round_ano_score_p[round, idx] = ano_score_p
                # multi_round_ano_score_n[round, idx] = ano_score_n

            pbar_test.update(1)
    return multi_round_ano_score, ano_label,first_epoch_loss,last_epoch_loss

if __name__ == '__main__':
    subgraph_sizes = [2,3,4,5]
    datasets = ['cora','citeseer','pubmed','ACM','BlogCatalog','Flickr']
    ratios = {'cora':0.051,'citeseer':0.045,'pubmed':0.025,'ACM':0.036,'BlogCatalog':0.058,
              'Flickr':0.059}
    for dataset in datasets:
        anomaly_scores = pd.DataFrame()
        ratio = ratios[dataset]
        output_file_0 = open(f'Output/CoLA-{dataset}.txt', 'w')
        first_line = 'seed, subgraph_size, AUC, CST, CST_equal, CST_base, loss_start_epoch, loss_end_epoch'
        print(first_line, file=output_file_0)
        for subgraph_size in subgraph_sizes:
            args = padding_args(subgraph_size=subgraph_size,Dataset=dataset).parse_args()
            print("Alg:CoLA")
            print(f"seed:{args.seed}")
            print(f"dataset:{dataset}")
            print(f"subgraph_size:{subgraph_size}")
            start_time = time.time()
            
            ## Current hyperparameter
            current_hp = f'{args.seed},{subgraph_size},'

            

            output_file = open(f'Output/details/{args.dataset}/CoLA-{args.subgraph_size}-{args.dataset}-output.txt', 'w')
            multi_round_ano_score,ano_label,first_epoch_loss,last_epoch_loss = train_test_model(args,output_file)

            ano_score_final = np.mean(multi_round_ano_score, axis=0)

            # Compute the statistic for the given list
            ccs = compute_custom_statistic(ano_score_final,ratio)
            computed_statistic = '{:.5f}'.format(ccs)
            computed_equal = '{:.5f}'.format(compute_custom_statistic_equal(ano_score_final,ratio))
            computed_base = '{:.5f}'.format(compute_custom_statistic_base(ano_score_final,ratio))

            print('\n Contrast Score Margin:{:.5f}'.format(ccs))
            print('\n Contrast Score Margin:{:.5f}'.format(ccs), file=output_file)

            CSTs = f'{computed_statistic},{computed_equal},{computed_base},'
            # ano_score_final_p = np.mean(multi_round_ano_score_p, axis=0)
            # ano_score_final_n = np.mean(multi_round_ano_score_n, axis=0)
            auc = roc_auc_score(ano_label, ano_score_final)
            ## Save auc value
            AUC = '{:.4f},'.format(auc)

            print('AUC:{:.4f}'.format(auc))
            print('AUC:{:.4f}'.format(auc), file=output_file)

            # Seed,HP,auc,CSTs,start_loss,end_loss
            output_line = current_hp+AUC+CSTs+f'{first_epoch_loss},{last_epoch_loss}'
            print(output_line,file=output_file_0)
            # End the timer
            end_time = time.time()

            # Calculate the total running time
            running_time = end_time - start_time

            # Convert the running time to hours, minutes, and seconds
            hours = int(running_time // 3600)
            minutes = int((running_time % 3600) // 60)
            seconds = running_time % 60

            # Formatting the output as hour:min:sec
            formatted_time = "{:02d}:{:02d}:{:06.3f}".format(hours, minutes, seconds)

            # print("\n run time: ")
            # print(formatted_time)
            print("\n run time: ", file=output_file)
            print(formatted_time, file=output_file)
            col_name = f'subgraph_size={subgraph_size}'
            anomaly_scores[col_name] = ano_score_final
            # 关闭文件
            
            output_file.close()

        csv_filename = f'Output/anomaly_scores_{dataset}.csv'
        anomaly_scores.to_csv(csv_filename, index=False)
        output_file_0.close()



