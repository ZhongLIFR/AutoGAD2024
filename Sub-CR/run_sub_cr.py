"""
Author: z.li@liacs.leidenuni.nl
repurposed based on the source code of Sub-CR: https://github.com/Zjer12/Sub
"""

import numpy as np
from numpy.core.fromnumeric import shape
import scipy.sparse as sp
import torch
import torch.nn as nn
from aug import *
from model import *
from utils import *
from aug import *
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import MinMaxScaler

import random
import os
import dgl

import argparse
from tqdm import tqdm

# =============================================================================
# Step 1: Set argument
# =============================================================================
parser = argparse.ArgumentParser(description='AutoGAD')
parser.add_argument('--dataset', type=str,default='cora')  # 'BlogCatalog'  'Flickr'   'cora'  'citeseer'  'pubmed'
parser.add_argument('--lr', type=float)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=2)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--num_epoch', type=int)
parser.add_argument('--drop_prob', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--subgraph_size', type=int, default=4)
parser.add_argument('--readout', type=str, default='avg')  # max min avg  weighted_sum
parser.add_argument('--auc_test_rounds', type=int, default=299) ##number of rounds to compute anomaly score
parser.add_argument('--negsamp_ratio', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--gamma', type=float, default=0.60) ##balance-parameter of L_{con} and L_{res}, default = 0.6
parser.add_argument('--alpha', type=float, default=0.01) ##teleport probability in graph difussion, default = 0.01

args = parser.parse_args()

if args.lr is None:
    if args.dataset in ['cora', 'citeseer', 'pubmed', 'Flickr']:
        args.lr = 1e-3
    elif args.dataset == 'BlogCatalog':
        args.lr = 3e-3

if args.num_epoch is None:
    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        args.num_epoch = 100
    elif args.dataset in ['BlogCatalog', 'Flickr']:
        args.num_epoch = 400

batch_size = args.batch_size
subgraph_size = args.subgraph_size

print('Dataset: ', args.dataset)

# =============================================================================
# Step 2: Set devices and random seeds
# =============================================================================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

## Set random seed
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

# =============================================================================
# Step 3: Load and preprocess data
# =============================================================================
##Load data
adj, features, labels, idx_train, idx_val, \
idx_test, ano_label, str_ano_label, attr_ano_label = load_mat(args.dataset)

#Generate or Load diffusion data
diff=gdc(adj,alpha=args.alpha,eps=0.0001)
np.save('cora',diff)  #'cora' can be changed to 'BlogCatalog' 'cite' 'Flickr'...

##Local view (Raw data)
##----------------------------------------------------------------------------

##Process data
dgl_graph = adj_to_dgl_graph(adj) ##generate graph

#Process adj matrix
adj = normalize_adj(adj)
adj = (adj + sp.eye(adj.shape[0])).todense()
adj = torch.FloatTensor(adj[np.newaxis])

#Process node features (keep Raw features and Processed features)
raw_feature=features.todense()
raw_feature = torch.FloatTensor(raw_feature[np.newaxis])

features, _ = preprocess_features(features)
nb_nodes = features.shape[0] ##number of nodes
ft_size = features.shape[1]  ##dimension of node features
features = torch.FloatTensor(features[np.newaxis])

##Global view (diffusion data)
##----------------------------------------------------------------------------

#Process data (only adj matrix)
b_adj = sp.csr_matrix(diff)
b_adj = (b_adj + sp.eye(b_adj.shape[0])).todense()
b_adj = torch.FloatTensor(b_adj[np.newaxis])

# =============================================================================
# Step 4: Initialize model and optimiser
# =============================================================================
model = Model(args.subgraph_size, ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout, args.dropout)

optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if torch.cuda.is_available():
    print('Using CUDA')
    model.to(device)
    features = features.to(device)
    raw_feature = raw_feature.to(device)
    adj = adj.to(device)
    b_adj = b_adj.to(device)

if torch.cuda.is_available():
    b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).to(device))
else:
    b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]))
xent = nn.CrossEntropyLoss()
cnt_wait = 0
best = 1e9
best_t = 0
batch_num = nb_nodes // batch_size + 1

added_adj_zero_row = torch.zeros((nb_nodes, 1, subgraph_size))
added_adj_zero_col = torch.zeros((nb_nodes, subgraph_size + 1, 1))
added_adj_zero_col[:, -1, :] = 1.
added_feat_zero_row = torch.zeros((nb_nodes, 1, ft_size))
if torch.cuda.is_available():
    added_adj_zero_row = added_adj_zero_row.to(device)
    added_adj_zero_col = added_adj_zero_col.to(device)
    added_feat_zero_row = added_feat_zero_row.to(device)
mse_loss = nn.MSELoss(reduction='mean')

# =============================================================================
# Step 5: Train model
# =============================================================================
with tqdm(total=args.num_epoch) as pbar:
    pbar.set_description('Training')
    
    for epoch in range(args.num_epoch):

        loss_full_batch = torch.zeros((nb_nodes, 1))
        if torch.cuda.is_available():
            loss_full_batch = loss_full_batch.to(device)

        model.train()

        all_idx = list(range(nb_nodes))

        random.shuffle(all_idx)
        total_loss = 0.
        
        ##generate subgraph
        subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)
        
        epoch_count = 0
        
        for batch_idx in range(batch_num):

            optimiser.zero_grad()

            is_final_batch = (batch_idx == (batch_num))

            if not is_final_batch:
                idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            else:
                idx = all_idx[batch_idx * batch_size:]

            cur_batch_size = len(idx)

            lbl = torch.unsqueeze(torch.cat((torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio))), 1)
            
            ## for cpu
            added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size))
            added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1))
            added_adj_zero_col[:, -1, :] = 1.
            added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))

            ## for gpu
            if torch.cuda.is_available():
                lbl = lbl.to(device)
                added_adj_zero_row = added_adj_zero_row.to(device)
                added_adj_zero_col = added_adj_zero_col.to(device)
                added_feat_zero_row = added_feat_zero_row.to(device)

            ## containers to store matrices
            ba = []
            bf = []
            br = []
            raw=[]
            
            for i in idx:
                
                ##local view adjcency
                cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                ba.append(cur_adj)
                
                ##global view adjcency
                cur_adj_r = b_adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                br.append(cur_adj_r)
                
                ##raw node features          
                raw_f=raw_feature[:, subgraphs[i], :]
                raw.append(raw_f)
                
                ##node features to reconstruct
                cur_feat = features[:, subgraphs[i], :]
                bf.append(cur_feat)
                

            ##What is ba, br and bf?
            ##local view adjcency
            ba = torch.cat(ba)
            ba = torch.cat((ba, added_adj_zero_row), dim=1)
            ba = torch.cat((ba, added_adj_zero_col), dim=2)

            ##global view adjcency
            br = torch.cat(br)
            br = torch.cat((br, added_adj_zero_row), dim=1)
            br = torch.cat((br, added_adj_zero_col), dim=2)

            ##raw node features 
            raw = torch.cat(raw)
            raw = torch.cat((raw[:, :-1, :], added_feat_zero_row, raw[:, -1:, :]), dim=1)

            ##node features to reconstruct
            bf = torch.cat(bf)
            bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)


            # print("\n------")
            # print(bf.size())
            # print(ba.size())
            # print(br.size())
            # print(raw.size())
            # print("\n------")
            
            ##Local view            
            now1, logits = model(bf, ba, raw)
            
            ##Global view
            now2, logits2 = model(bf, br, raw)
            
            batch = now1.shape[0]
            
            # print("\n------")
            # print(now1.size())
            # print(now2.size())
            # print(raw[:, -1, :].size())
            # print("\n------")
            
            ##Compute Attribute Reconstruction Loss (Local view + Global view)
            ##-----------------------------------------------------------------
            loss_re_local = mse_loss(now1, raw[:, -1, :])
            loss_re_global = mse_loss(now2, raw[:, -1, :])
            loss_re= 0.5 * (loss_re_local + loss_re_global)
            
            
            ##Compute Intra-View Contrastive Loss (Local view + Global view)
            ##-----------------------------------------------------------------
            #Local view
            loss_bce_local = b_xent(logits, lbl)
            
            #Global view
            loss_bce_global = b_xent(logits2, lbl)
            
            #Add local view and global view
            loss_bce = 0.5 * (loss_bce_local + loss_bce_global)
            loss_con_intra = torch.mean(loss_bce)
                  
            ##Compute Inter-View Contrastive Loss (Local view + Global view)
            ##-----------------------------------------------------------------
            #Local view
            h_local = F.normalize(logits[:batch, :], dim=1, p=2)
            
            #Global view
            h_global = F.normalize(logits2[:batch, :], dim=1, p=2)
            
            #Compare local view and global view
            loss_con_inter = 2 - 2 * (h_local * h_global).sum(dim=-1).mean()
            
            ##Combine Attribute Reconstruction Loss, Intra-View Contrastive Loss, and Inter-View Contrastive Loss
            ##-----------------------------------------------------------------
            loss = args.gamma*loss_re + loss_con_intra + loss_con_inter 

            ##Bp gradients
            loss.backward()   
            optimiser.step()

            ##Compute loss values
            loss = loss.detach().cpu().numpy()
            total_loss += loss
            epoch_count = epoch_count + 1 ##used to compute mean loss value

        mean_loss = total_loss/epoch_count

        ##store the best model so far
        if mean_loss < best:
            best = mean_loss
            best_t = epoch
            cnt_wait = 0 ##use to store patience epochs
            torch.save(model.state_dict(), 'best_model.pkl')  #use this model

        else:
            cnt_wait += 1

        pbar.update(1)

# =============================================================================
# Step 6: Test model
# =============================================================================
print('Loading {}th epoch'.format(best_t))
model.load_state_dict(torch.load('best_model.pkl'))

##vector to store anomaly scores of multiple rounds
multi_round_ano_score = np.zeros((args.auc_test_rounds, nb_nodes))

with tqdm(total=args.auc_test_rounds) as pbar_test:
    pbar_test.set_description('Testing')
    
    ##using multiple rounds to better estimate the anomaly scores
    for round in range(args.auc_test_rounds):

        all_idx = list(range(nb_nodes))
        random.shuffle(all_idx)

        ##generate subgraphs
        subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)
        
        
        for batch_idx in range(batch_num):

            optimiser.zero_grad()

            is_final_batch = (batch_idx == (batch_num - 1))

            ##batch可能不是batchzise=300的整数倍
            if not is_final_batch:
                idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            else:
                idx = all_idx[batch_idx * batch_size:]

            cur_batch_size = len(idx)

            ##containers to store matrices
            
            
            ##subgraph for cpu
            added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size))
            added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1))
            added_adj_zero_col[:, -1, :] = 1.
            added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))

            ##subgraph for gpu
            if torch.cuda.is_available():
                added_adj_zero_row = added_adj_zero_row.to(device)
                added_adj_zero_col = added_adj_zero_col.to(device)
                added_feat_zero_row = added_feat_zero_row.to(device)

            ##COntainers to store vectors 
            ba = []
            br = []
            raw=[]
            bf = []
            
            for i in idx:
                
                ##local view adjcency
                cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                ba.append(cur_adj)
                
                ##global view adjcency
                cur_adj2 = b_adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                br.append(cur_adj2)
                          
                ##raw node features 
                raw_f = raw_feature[:, subgraphs[i], :]
                raw.append(raw_f)
                
                ##node features to reconstruct
                cur_feat = features[:, subgraphs[i], :]
                bf.append(cur_feat)
                
            ##local view adjcency
            ba = torch.cat(ba)
            ba = torch.cat((ba, added_adj_zero_row), dim=1)
            ba = torch.cat((ba, added_adj_zero_col), dim=2)
            
            ##global view adjcency
            br = torch.cat(br)
            br = torch.cat((br, added_adj_zero_row), dim=1)
            br = torch.cat((br, added_adj_zero_col), dim=2)
 
            ##raw node features
            raw = torch.cat(raw)
            raw = torch.cat((raw[:, :-1, :], added_feat_zero_row, raw[:, -1:, :]), dim=1)
            
            ##node features to reconstruct
            bf = torch.cat(bf)
            bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)


            with torch.no_grad(): ##test procedure does not use bp
            
                ##Local view 
                now1, logits = model(bf, ba,raw)

                logits = torch.squeeze(logits)
                logits = torch.sigmoid(logits)

                ##Global view 
                now2, logits2 = model(bf, br,raw)
                
                logits2 = torch.squeeze(logits2)
                logits2 = torch.sigmoid(logits2)
                
    
            ##anomaly score for contrastive learning
            ##--------------------------------------
            ##Local view
            con_score_local = - (logits[:cur_batch_size] - logits[cur_batch_size:]).cpu().numpy()
            
            ##Global view
            con_score_global = - (logits2[:cur_batch_size] - logits2[cur_batch_size:]).cpu().numpy()
            
            ##Combine local and global scores
            score_con= 0.5*(con_score_local + con_score_global)
            
            ##Scale scores 
            scaler_con = MinMaxScaler()
            ano_score_co = scaler_con.fit_transform(score_con.reshape(-1, 1)).reshape(-1)
        
            ##anomaly score for reconstruction
            ##--------------------------------------       
            pdist = nn.PairwiseDistance(p=2)
            
            ##Local view
            re_score_local = pdist(now1, raw[:, -1, :])
            
            ##Global view
            re_score_global = pdist(now2, raw[:, -1, :])
            
            ##Combine local and global scores
            score_re = 0.5*(re_score_local + re_score_global)
            score_re = score_re.cpu().numpy()
            
            ##Scale scores 
            scaler_re = MinMaxScaler()
            ano_score_re = scaler_re.fit_transform(score_re.reshape(-1, 1)).reshape(-1)
            
            ##combine contrastive learning anomaly score  and reconstruction anomaly score
            ##-------------------------------------
            ano_scores = ano_score_co + args.gamma*ano_score_re
            
            ##store results for current round
            multi_round_ano_score[round, idx] = ano_scores

        pbar_test.update(1)

##take the average score of multiple round as final anomaly score 
ano_score_final = np.mean(multi_round_ano_score, axis=0)

##compute auc score
auc = roc_auc_score(ano_label, ano_score_final)
print('\n AUC:{:.4f}'.format(auc))

