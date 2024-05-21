from model import Model
from utils import *
from sklearn.metrics import roc_auc_score
import random
import os
import dgl
import argparse
from tqdm import tqdm
from CSTs import compute_custom_statistic,compute_custom_statistic_base,compute_custom_statistic_equal
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['OMP_NUM_THREADS'] = '1'
def padding_args(alpha = 0.9, subgraph_size = 4, dataset = 'pubmed'):
    parser = argparse.ArgumentParser(description='ANEMONE')
    parser.add_argument('--expid', type=int)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default=dataset) ##from 'cora', 'citeseer', 'pubmed'
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=300)
    parser.add_argument('--subgraph_size', type=int, default=subgraph_size)  ##4 for cora and citeseer
    parser.add_argument('--readout', type=str, default='avg')
    parser.add_argument('--auc_test_rounds', type=int, default=300)
    parser.add_argument('--negsamp_ratio_patch', type=int, default=1)
    parser.add_argument('--negsamp_ratio_context', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=alpha, help='how much context-level involves') ##0.90 for cora, citeseer and pubmed
    args = parser.parse_args()
    return args
# =============================================================================
# Define a function to compute internal metric score
# =============================================================================

def compute_custom_statistic(data,ratio = 0.51):
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

    batch_size = args.batch_size
    subgraph_size = args.subgraph_size

    adj, features, labels, idx_train, idx_val,\
    idx_test, ano_label, str_ano_label, attr_ano_label = load_mat(args.dataset)

    features, _ = preprocess_features(features)
    dgl_graph = adj_to_dgl_graph(adj)

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = labels.shape[1]

    adj = normalize_adj(adj)
    adj = (adj + sp.eye(adj.shape[0])).todense()

    features = torch.FloatTensor(features[np.newaxis]).to(device)
    adj = torch.FloatTensor(adj[np.newaxis]).to(device)
    labels = torch.FloatTensor(labels[np.newaxis]).to(device)
    idx_train = torch.LongTensor(idx_train).to(device)
    idx_val = torch.LongTensor(idx_val).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)

    all_auc = []
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建模型保存路径
    model_save_path = os.path.join(script_dir, 'checkpoints', 'exp_{}.pkl'.format(args.expid))

    # 确保checkpoints目录存在
    if not os.path.exists(os.path.join(script_dir, 'checkpoints')):
        os.makedirs(os.path.join(script_dir, 'checkpoints'))
    for run in range(1):
        seed = args.seed
        print('\n# Run:{} with random seed:{}'.format(run, seed), flush=True)
        dgl.random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
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
        first_epoch_loss = 0
        last_epoch_loss = 0
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
                bf = []
                added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size)).to(device)
                added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1)).to(device)
                added_adj_zero_col[:, -1, :] = 1.
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size)).to(device)

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

                logits_1, logits_2 = model(bf, ba)

                # Context-level
                loss_all_1 = b_xent_context(logits_1, lbl_context)
                loss_1 = torch.mean(loss_all_1)

                # Patch-level
                loss_all_2 = b_xent_patch(logits_2, lbl_patch)
                loss_2 = torch.mean(loss_all_2)

                loss = args.alpha * loss_1 + (1 - args.alpha) * loss_2

                loss.backward()
                optimiser.step()

                loss = loss.detach().cpu().numpy()
                if not is_final_batch:
                    total_loss += loss

            mean_loss = (total_loss * batch_size + loss * cur_batch_size) / nb_nodes
            
            if epoch == 0:
                first_epoch_loss = mean_loss
            if epoch == args.num_epoch-1:
                last_epoch_loss = mean_loss
            # 保存模型
            if mean_loss < best:
                best = mean_loss
                best_t = epoch
                cnt_wait = 0
                torch.save(model.state_dict(), model_save_path.format(args.expid))
            else:
                cnt_wait += 1

            if cnt_wait == args.patience:
                print('Early stopping!', flush=True)
                break

            print('Epoch:{} Loss:{:.8f}'.format(epoch, mean_loss), flush=True)
            print('Epoch:{} Loss:{:.8f}'.format(epoch, mean_loss), file=output_file)

        # Testing
        print('Loading {}th epoch'.format(best_t), flush=True)
        print('Loading {}th epoch'.format(best_t), file=output_file)
        # model.load_state_dict(torch.load('checkpoints/exp_{}.pkl'.format(args.expid)))
        model.load_state_dict(torch.load(model_save_path.format(args.expid)))
        multi_round_ano_score = np.zeros((args.auc_test_rounds, nb_nodes))
        print('Testing AUC!', flush=True)
        print('Testing AUC!', file=output_file)
        with tqdm(total=args.auc_test_rounds, desc='Testing') as pbar_test:
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
                    added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size)).to(device)
                    added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1)).to(device)
                    added_adj_zero_col[:, -1, :] = 1.
                    added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size)).to(device)
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
                        test_logits_1, test_logits_2 = model(bf, ba)
                        test_logits_1 = torch.sigmoid(torch.squeeze(test_logits_1))
                        test_logits_2 = torch.sigmoid(torch.squeeze(test_logits_2))

                    if args.alpha != 1.0 and args.alpha != 0.0:
                        if args.negsamp_ratio_context == 1 and args.negsamp_ratio_patch == 1:
                            ano_score_1 = - (test_logits_1[:cur_batch_size] - test_logits_1[cur_batch_size:]).cpu().numpy()
                            ano_score_2 = - (test_logits_2[:cur_batch_size] - test_logits_2[cur_batch_size:]).cpu().numpy()
                        else:
                            ano_score_1 = - (test_logits_1[:cur_batch_size] - torch.mean(test_logits_1[cur_batch_size:].view(
                                cur_batch_size, args.negsamp_ratio_context), dim=1)).cpu().numpy()  # context
                            ano_score_2 = - (test_logits_2[:cur_batch_size] - torch.mean(test_logits_2[cur_batch_size:].view(
                                cur_batch_size, args.negsamp_ratio_patch), dim=1)).cpu().numpy()  # patch
                        ano_score = args.alpha * ano_score_1 + (1 - args.alpha) * ano_score_2
                    elif args.alpha == 1.0:
                        if args.negsamp_ratio_context == 1:
                            ano_score = - (test_logits_1[:cur_batch_size] - test_logits_1[cur_batch_size:]).cpu().numpy()
                        else:
                            ano_score = - (test_logits_1[:cur_batch_size] - torch.mean(test_logits_1[cur_batch_size:].view(
                                    cur_batch_size, args.negsamp_ratio_context), dim=1)).cpu().numpy()  # context
                    elif args.alpha == 0.0:
                        if args.negsamp_ratio_patch == 1:
                            ano_score = - (test_logits_2[:cur_batch_size] - test_logits_2[cur_batch_size:]).cpu().numpy()
                        else:
                            ano_score = - (test_logits_2[:cur_batch_size] - torch.mean(test_logits_2[cur_batch_size:].view(
                                    cur_batch_size, args.negsamp_ratio_patch), dim=1)).cpu().numpy()  # patch

                    multi_round_ano_score[round, idx] = ano_score
                pbar_test.update(1)
        ano_score_final = np.mean(multi_round_ano_score, axis=0) + np.std(multi_round_ano_score, axis=0)
        auc = roc_auc_score(ano_label, ano_score_final)
        all_auc.append(auc)
        print('Testing AUC:{:.4f}'.format(auc), flush=True)
        print('Testing AUC:{:.4f}'.format(auc), file=output_file)
        # Compute the statistic for the given list
        computed_statistic = compute_custom_statistic(ano_score_final,ratio)
        print('\n Contrast Score Margin:{:.5f}'.format(computed_statistic))
        print('\n Contrast Score Margin:{:.5f}'.format(computed_statistic),file=output_file)

    print('\n==============================')
    print(all_auc)
    print('FINAL TESTING AUC:{:.4f}'.format(np.mean(all_auc)))
    print('==============================')
    print('\n==============================',file=output_file)
    print(all_auc,file=output_file)
    print('FINAL TESTING AUC:{:.4f}'.format(np.mean(all_auc)),file=output_file)
    print('==============================',file=output_file)

    return multi_round_ano_score, ano_label,first_epoch_loss,last_epoch_loss

import time




if __name__ == '__main__':
    datasets = ['cora','citeseer','pubmed','ACM','BlogCatalog','Flickr']
    ratios = {'cora':0.051,'citeseer':0.045,'pubmed':0.025,'ACM':0.036,'BlogCatalog':0.058,
              'Flickr':0.059}
    subgraph_sizes=[2,3,4,5]
    alphas=[0, 0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99,1.0]
    for dataset in datasets:
        anomaly_scores = pd.DataFrame()
        ratio = ratios[dataset]
        output_file_0 = open(f'Output/ANEMONE-{dataset}.txt', 'w')
        first_line = 'seed,subgraph_size,alpha,AUC,CST,CST_equal,CST_base,loss_start_epoch,loss_end_epoch'
        print(first_line,file=output_file_0)
        for subgraph_size in subgraph_sizes:
            for alpha in alphas:
                args = padding_args(dataset=dataset,subgraph_size=subgraph_size,alpha=alpha)
                print("Alg:ANEMONE")
                print(f"seed:{args.seed}")
                print(f"dataset:{dataset}")
                print(f"subgraph_size:{subgraph_size}")
                print(f"alpha:{alpha}")
                # Start the timer
                start_time = time.time()
                # Padding args

                
                current_hp = f'{args.seed},{subgraph_size},{alpha},'

                output_file = open(f'Output/details/{dataset}/ANEMONE-{args.subgraph_size}-{args.alpha}-{args.dataset}-output.txt', 'w')
                # train_test_model(args,output_file,ratio)
                multi_round_ano_score,ano_label,first_epoch_loss,last_epoch_loss = train_test_model(args,output_file,ratio)

                ano_score_final = np.mean(multi_round_ano_score, axis=0)
                
                ccs = compute_custom_statistic(ano_score_final,ratio)
                computed_statistic = '{:.5f}'.format(ccs)
                computed_equal = '{:.5f}'.format(compute_custom_statistic_equal(ano_score_final,ratio))
                computed_base = '{:.5f}'.format(compute_custom_statistic_base(ano_score_final,ratio))

                print('\n Contrast Score Margin:{:.5f}'.format(ccs))
                print('\n Contrast Score Margin:{:.5f}'.format(ccs), file=output_file)

                CSTs = f'{computed_statistic},{computed_equal},{computed_base},'

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
                col_name = f'subgraph_size={subgraph_size}_alpha={alpha}'
                anomaly_scores[col_name] = ano_score_final
                print("\n run time: ")
                print("\n run time: ",file=output_file)
                print(formatted_time)
                print(formatted_time,file=output_file)

        csv_filename = f'Output/anomaly_scores_{dataset}.csv'
        anomaly_scores.to_csv(csv_filename, index=False)
        output_file_0.close()