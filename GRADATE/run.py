from model import Model
from utils import *
from sklearn.metrics import roc_auc_score
import random
import os
import dgl
import argparse

from tqdm import tqdm

import torch.nn.functional as F
import pandas as pd


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['OMP_NUM_THREADS'] = '1'
def padding_args(alpha, beta, gamma,dataset,seed=2):
    parser = argparse.ArgumentParser(description='GRADATE')
    parser.add_argument('--seed', type=int, default=seed)
    parser.add_argument('--expid', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default=dataset) ## 'cora', 'citeseer'
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--num_epoch', type=int, default=400) ##400 for all
    parser.add_argument('--batch_size', type=int, default=300)
    parser.add_argument('--subgraph_size', type=int, default=4)
    parser.add_argument('--readout', type=str, default='avg')
    parser.add_argument('--auc_test_rounds', type=int, default=300)
    parser.add_argument('--negsamp_ratio_patch', type=int, default=6)
    parser.add_argument('--negsamp_ratio_context', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=alpha, help='balance the first view and the second view')##0.90 for 'cora'
    parser.add_argument('--beta', type=float, default=beta, help='balance between node-node and node-sub') ##0.3 for all
    parser.add_argument('--gamma', type=float, default=gamma, help='how much the sub-sub contrast involves') ##0.1 for all
    parser.add_argument('--prop', type=float, default=0.20, help='the proportion of modified edges') ##0.2 for all


    args = parser.parse_args()

    return args



def compute_custom_statistic_base(data, ratio=0.05):
    # Sort the list in decreasing order
    sorted_data = sorted(data, reverse=True)

    # Determine the index for the top 5% of the elements
    # Ensure at least one element in A for small lists
    top_5_percent_index = max(1, int(len(sorted_data) * ratio))

    # Split the data into A (top 5%) and B (the rest)
    A = sorted_data[:top_5_percent_index]
    B = sorted_data[top_5_percent_index:top_5_percent_index*2]

    # Compute means of A and B
    mean_A = np.mean(A)
    mean_B = np.mean(B)

    # Compute variances of A and B
    var_A = np.var(A, ddof=1)  # ddof=1 for sample variance
    var_B = np.var(B, ddof=1)

    # Compute the required statistic
    statistic = mean_A + (mean_A - mean_B) / np.sqrt(var_A + var_B)

    return statistic

def compute_custom_statistic_equal(data, ratio=0.05):
    # Sort the list in decreasing order
    sorted_data = sorted(data, reverse=True)

    # Determine the index for the top 5% of the elements
    # Ensure at least one element in A for small lists
    top_5_percent_index = max(1, int(len(sorted_data) * ratio))

    # Split the data into A (top 5%) and B (the rest)
    A = sorted_data[:top_5_percent_index]
    B = sorted_data[top_5_percent_index:top_5_percent_index*2]

    # Compute means of A and B
    mean_A = np.mean(A)
    mean_B = np.mean(B)

    # Compute variances of A and B
    var_A = np.var(A, ddof=1)  # ddof=1 for sample variance
    var_B = np.var(B, ddof=1)

    # Compute the required statistic
    statistic = (mean_A - mean_B) / np.sqrt(var_A + var_B)

    return statistic


def compute_custom_statistic(data, ratio=0.05):
    # Sort the list in decreasing order
    sorted_data = sorted(data, reverse=True)

    # Determine the index for the top 5% of the elements
    # Ensure at least one element in A for small lists
    top_5_percent_index = max(1, int(len(sorted_data) * ratio))

    # Split the data into A (top 5%) and B (the rest)
    A = sorted_data[:top_5_percent_index]
    B = sorted_data[top_5_percent_index:]

    # Compute means of A and B
    mean_A = np.mean(A)
    mean_B = np.mean(B)

    # Compute variances of A and B
    var_A = np.var(A, ddof=1)  # ddof=1 for sample variance
    var_B = np.var(B, ddof=1)

    # Compute the required statistic
    statistic = (mean_A - mean_B) / np.sqrt(var_A + var_B)

    return statistic
def train_test_model(args,output_file,ratio):
    print('Dataset: {}'.format(args.dataset), flush=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # for run in range(args.runs):

    seed = args.seed
    random.seed(seed)


    batch_size = args.batch_size
    subgraph_size = args.subgraph_size

    adj, features, labels, idx_train, idx_val,\
    idx_test, ano_label, str_ano_label, attr_ano_label = load_mat(args.dataset)



    features, _ = preprocess_features(features)
    dgl_graph = adj_to_dgl_graph(adj)

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = labels.shape[1]

    # graph data argumentation
    adj_edge_modification = aug_random_edge(adj, args.prop)
    adj = normalize_adj(adj)
    adj = (adj + sp.eye(adj.shape[0])).todense()
    adj_hat = normalize_adj(adj_edge_modification)
    adj_hat = (adj_hat + sp.eye(adj_hat.shape[0])).todense()

    features = torch.FloatTensor(features[np.newaxis]).to(device)
    adj = torch.FloatTensor(adj[np.newaxis]).to(device)
    adj_hat = torch.FloatTensor(adj_hat[np.newaxis]).to(device)
    labels = torch.FloatTensor(labels[np.newaxis]).to(device)
    idx_train = torch.LongTensor(idx_train).to(device)
    idx_val = torch.LongTensor(idx_val).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)

    all_auc = []


    print('\n# Run:{} with random seed:{}'.format(args.runs, seed), flush=True)
    dgl.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    model = Model(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio_patch, args.negsamp_ratio_context,
                    args.readout).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    b_xent_patch = nn.BCEWithLogitsLoss(reduction='none',
                                        pos_weight=torch.tensor([args.negsamp_ratio_patch]).to(device))
    b_xent_context = nn.BCEWithLogitsLoss(reduction='none',
                                        pos_weight=torch.tensor([args.negsamp_ratio_context]).to(device))

    cnt_wait = 0
    best = 1e9
    best_t = 0
    batch_num = nb_nodes // batch_size + 1

    for epoch in range(args.num_epoch):

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

            lbl_patch = torch.unsqueeze(torch.cat(
                (torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio_patch))), 1).to(device)

            lbl_context = torch.unsqueeze(torch.cat(
                (torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio_context))), 1).to(device)

            ba = []
            ba_hat = []
            bf = []
            added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size)).to(device)
            added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1)).to(device)
            added_adj_zero_col[:, -1, :] = 1.
            added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size)).to(device)

            for i in idx:
                cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                cur_adj_hat = adj_hat[:, subgraphs[i], :][:, :, subgraphs[i]]
                cur_feat = features[:, subgraphs[i], :]
                ba.append(cur_adj)
                ba_hat.append(cur_adj_hat)
                bf.append(cur_feat)

            ba = torch.cat(ba)
            ba = torch.cat((ba, added_adj_zero_row), dim=1)
            ba = torch.cat((ba, added_adj_zero_col), dim=2)
            ba_hat = torch.cat(ba_hat)
            ba_hat = torch.cat((ba_hat, added_adj_zero_row), dim=1)
            ba_hat = torch.cat((ba_hat, added_adj_zero_col), dim=2)
            bf = torch.cat(bf)
            bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)

            logits_1, logits_2, subgraph_embed, node_embed = model(bf, ba)
            logits_1_hat, logits_2_hat,  subgraph_embed_hat, node_embed_hat = model(bf, ba_hat)

            #subgraph-subgraph contrast loss
            subgraph_embed = F.normalize(subgraph_embed, dim=1, p=2)
            subgraph_embed_hat = F.normalize(subgraph_embed_hat, dim=1, p=2)
            sim_matrix_one = torch.matmul(subgraph_embed, subgraph_embed_hat.t())
            sim_matrix_two = torch.matmul(subgraph_embed, subgraph_embed.t())
            sim_matrix_three = torch.matmul(subgraph_embed_hat, subgraph_embed_hat.t())
            temperature = 1.0
            sim_matrix_one_exp = torch.exp(sim_matrix_one / temperature)
            sim_matrix_two_exp = torch.exp(sim_matrix_two / temperature)
            sim_matrix_three_exp = torch.exp(sim_matrix_three / temperature)
            nega_list = np.arange(0, cur_batch_size - 1, 1)
            nega_list = np.insert(nega_list, 0, cur_batch_size - 1)
            sim_row_sum = sim_matrix_one_exp[:, nega_list] + sim_matrix_two_exp[:, nega_list] + sim_matrix_three_exp[:, nega_list]
            sim_row_sum = torch.diagonal(sim_row_sum)
            sim_diag = torch.diagonal(sim_matrix_one)
            sim_diag_exp = torch.exp(sim_diag / temperature)
            NCE_loss = -torch.log(sim_diag_exp / (sim_row_sum))
            NCE_loss = torch.mean(NCE_loss)



            loss_all_1 = b_xent_context(logits_1, lbl_context)
            loss_all_1_hat = b_xent_context(logits_1_hat, lbl_context)
            loss_1 = torch.mean(loss_all_1)
            loss_1_hat = torch.mean(loss_all_1_hat)

            loss_all_2 = b_xent_patch(logits_2, lbl_patch)
            loss_all_2_hat = b_xent_patch(logits_2_hat, lbl_patch)
            loss_2 = torch.mean(loss_all_2)
            loss_2_hat = torch.mean(loss_all_2_hat)

            loss_1 = args.alpha * loss_1 + (1 - args.alpha) * loss_1_hat #node-subgraph contrast loss
            loss_2 = args.alpha * loss_2 + (1 - args.alpha) * loss_2_hat #node-node contrast loss
            loss = args.beta * loss_1 + (1 - args.beta) * loss_2 + args.gamma * NCE_loss #total loss

            loss.backward()
            optimiser.step()

            loss = loss.detach().cpu().numpy()
            if not is_final_batch:
                total_loss += loss

        mean_loss = (total_loss * batch_size + loss * cur_batch_size) / nb_nodes
        if epoch == 0:
            first_loss_value = mean_loss
        if epoch == args.num_epoch - 1:
            last_loss_value = mean_loss

        if mean_loss < best:
            best = mean_loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), '{}.pkl'.format(args.dataset))
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!', flush=True)
            break

        print('Epoch:{} Loss:{:.8f}'.format(epoch, mean_loss), flush=True)
        print('Epoch:{} Loss:{:.8f}'.format(epoch, mean_loss), file=output_file)

    # Testing
    print('Loading {}th epoch'.format(best_t), flush=True)
    model.load_state_dict(torch.load('{}.pkl'.format(args.dataset)))
    multi_round_ano_score = np.zeros((args.auc_test_rounds, nb_nodes))
    print('Testing AUC!', flush=True)

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
                ba_hat = []
                bf = []
                added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size)).to(device)
                added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1)).to(device)
                added_adj_zero_col[:, -1, :] = 1.
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size)).to(device)
                for i in idx:
                    cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_adj_hat = adj_hat[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_feat = features[:, subgraphs[i], :]
                    ba.append(cur_adj)
                    ba_hat.append(cur_adj_hat)
                    bf.append(cur_feat)

                ba = torch.cat(ba)
                ba = torch.cat((ba, added_adj_zero_row), dim=1)
                ba = torch.cat((ba, added_adj_zero_col), dim=2)
                ba_hat = torch.cat(ba_hat)
                ba_hat = torch.cat((ba_hat, added_adj_zero_row), dim=1)
                ba_hat = torch.cat((ba_hat, added_adj_zero_col), dim=2)
                bf = torch.cat(bf)
                bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)

                with torch.no_grad():
                    test_logits_1, test_logits_2, _, _ = model(bf, ba)
                    test_logits_1_hat, test_logits_2_hat, _, _ = model(bf, ba_hat)
                    test_logits_1 = torch.sigmoid(torch.squeeze(test_logits_1))
                    test_logits_2 = torch.sigmoid(torch.squeeze(test_logits_2))
                    test_logits_1_hat = torch.sigmoid(torch.squeeze(test_logits_1_hat))
                    test_logits_2_hat = torch.sigmoid(torch.squeeze(test_logits_2_hat))


                    ano_score_1 = - (test_logits_1[:cur_batch_size] - torch.mean(test_logits_1[cur_batch_size:].view(
                        cur_batch_size, args.negsamp_ratio_context), dim=1)).cpu().numpy()
                    ano_score_1_hat = - (
                                test_logits_1_hat[:cur_batch_size] - torch.mean(test_logits_1_hat[cur_batch_size:].view(
                            cur_batch_size, args.negsamp_ratio_context), dim=1)).cpu().numpy()
                    ano_score_2 = - (test_logits_2[:cur_batch_size] - torch.mean(test_logits_2[cur_batch_size:].view(
                        cur_batch_size, args.negsamp_ratio_patch), dim=1)).cpu().numpy()
                    ano_score_2_hat = - (
                                test_logits_2_hat[:cur_batch_size] - torch.mean(test_logits_2_hat[cur_batch_size:].view(
                            cur_batch_size, args.negsamp_ratio_patch), dim=1)).cpu().numpy()
                    ano_score = args.beta * (args.alpha * ano_score_1 + (1 - args.alpha) * ano_score_1_hat)  + \
                                (1 - args.beta) * (args.alpha * ano_score_2 + (1 - args.alpha) * ano_score_2_hat)

                multi_round_ano_score[round, idx] = ano_score

            pbar_test.update(1)

        ano_score_final = np.mean(multi_round_ano_score, axis=0) + np.std(multi_round_ano_score, axis=0)
        
        # Compute the statistic for the given list
        # computed_statistic = compute_custom_statistic(ano_score_final)
        # print('\n Contrast Score Margin:{:.5f}'.format(computed_statistic))
        computed_statistic = compute_custom_statistic(ano_score_final,ratio)
        computed_statistic_equal = compute_custom_statistic_equal(ano_score_final,ratio)
        compute_custom_base = compute_custom_statistic_base(ano_score_final,ratio)
        print('\n Contrast Score Margin:{:.5f}'.format(computed_statistic))
        print('\n Contrast Score Margin Equal:{:.5f}'.format(computed_statistic_equal))
        print('\n Contrast Score Margin base:{:.5f}'.format(compute_custom_base))
        auc = roc_auc_score(ano_label, ano_score_final)
        all_auc.append(auc)
        print('Testing AUC:{:.4f}'.format(auc), flush=True)


    print('\n==============================')
    print(all_auc)
    print('FINAL TESTING AUC:{:.4f}'.format(np.mean(all_auc)))
    print('==============================')
    AUC = '{:.4f}'.format(np.mean(all_auc))
    cs = '{:.5f}'.format(computed_statistic)
    cse = '{:.5f}'.format(computed_statistic_equal)
    ccsb = '{:.5f}'.format(compute_custom_base)
    return AUC,cs,cse,ccsb,first_loss_value,last_loss_value,ano_score_final

import time

# Start the timer




if __name__ == '__main__':
    datasets = ['cora','citeseer']
    alphas = [0.9]
    betas = [0, 0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99,1.0]
    gammas = [0, 0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99,1.0]
    ratios = {'cora':0.051,'citeseer':0.045,'pubmed':0.025,'ACM':0.036,'BlogCatalog':0.058,
              'Flickr':0.059}
    seed_number = 2
    for dataset in datasets:
        anomaly_scores = pd.DataFrame()
        ratio = ratios[dataset]
        output_file_0 = open(f'Output/GRADATE-{dataset}.txt', 'w')
        first_line = 'seed,alpha,beta,gamma,AUC,CST,CST_equal,CST_base,loss_start_epoch,loss_end_epoch'
        print(first_line,file=output_file_0)
        for alpha in alphas:
            for beta in betas:
                for gamma in gammas:
                    print("Alg:GRADATE")
                    print(f"seed:{seed_number}")
                    print(f"dataset:{dataset}")
                    print(f"beta:{beta}")
                    print(f"gamma:{gamma}")

                    start_time = time.time()
                    current_hp = f'{seed_number},{alpha},{beta},{gamma},'
                    args = padding_args(alpha=alpha, beta=beta, gamma=gamma,dataset=dataset,seed=seed_number)
                    output_file = open(f'Output/details/{dataset}/GRADATE-{alpha}-{beta}-{gamma}-output.txt', 'w')
                    AUC,cs,cse,ccsb,first_loss_value,last_loss_value,ano_score_final = train_test_model(args=args,output_file=output_file,ratio=ratio)
                    results = f'{AUC},{cs},{cse},{ccsb},{first_loss_value},{last_loss_value}'
                    output_line = current_hp+results
                    print(output_line,file=output_file_0)

                    col_name = f'{seed_number},{alpha},{beta},{gamma}'
                    anomaly_scores[col_name] = ano_score_final
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

                    print("\n run time: ")
                    print("Alg:GRADATE")
                    print(f"beta:{beta}")
                    print(f"gamma:{gamma}")
                    print(formatted_time)
        csv_filename = f'Output/anomaly_scores_{dataset}.csv'
        anomaly_scores.to_csv(csv_filename, index=False)
        output_file_0.close()
