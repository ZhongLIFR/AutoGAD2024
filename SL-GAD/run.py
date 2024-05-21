from model import Model
from utils import *
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import random
import os
import dgl
import argparse
from tqdm import tqdm
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def padding_args(alpha=0.20, subgraph_size=4,dataset='cora',seed=2):
    parser = argparse.ArgumentParser(description='SL-GAD')
    parser.add_argument('--expid', type=int, default= 1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default=dataset) ##choose from 'cora', 'citeseer'(30m), 'pubmed'(1h)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--runs', type=int, default=2)
    parser.add_argument('--seed', type=int, default=seed)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--patience', type=int, default=400)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--drop_prob', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=300)
    parser.add_argument('--subgraph_size', type=int, default=subgraph_size) # 4 for all datasets
    parser.add_argument('--readout', type=str, default='avg')
    parser.add_argument('--auc_test_rounds', type=int, default=300)
    parser.add_argument('--negsamp_ratio', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=alpha) ##1.0 for cora
    parser.add_argument('--beta', type=float, default=0.6) ##0.6 for cora

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


import time

# Start the timer
def train_test_model(args,output_file,ratio):

    assert args.expid is not None, "experiment id needs to be assigned."
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    seeds = [i+1 for i in range(args.runs)]

    if args.lr is None:
        if args.dataset in ['cora','citeseer','pubmed','Flickr']:
            args.lr = 1e-3
        elif args.dataset == 'ACM':
            args.lr = 1e-3
        elif args.dataset == 'BlogCatalog':
            args.lr = 3e-3

    if args.num_epoch is None:
        if args.dataset in ['cora','citeseer','pubmed']:
            args.num_epoch = 100
        elif args.dataset in ['BlogCatalog','Flickr','ACM']:
            args.num_epoch = 400

    batch_size = args.batch_size
    subgraph_size = args.subgraph_size

    adj, features, labels, idx_train, idx_val,\
    idx_test, ano_label, str_ano_label, attr_ano_label = load_mat(args.dataset)

    raw_features = features.todense()
    features, _ = preprocess_features(features)

    dgl_graph = adj_to_dgl_graph(adj)

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = labels.shape[1]

    adj = normalize_adj(adj)
    adj = (adj + sp.eye(adj.shape[0])).todense()

    features = torch.FloatTensor(features[np.newaxis]).to(device)
    raw_features = torch.FloatTensor(raw_features[np.newaxis]).to(device)
    adj = torch.FloatTensor(adj[np.newaxis]).to(device)
    labels = torch.FloatTensor(labels[np.newaxis]).to(device)
    idx_train = torch.LongTensor(idx_train).to(device)
    idx_val = torch.LongTensor(idx_val).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)

    #####################
    all_auc = []
    for run in range(1):
        # setup seed
        seed = args.seed
        print('\n# Run:{}'.format(run), flush=True)
        dgl.random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['OMP_NUM_THREADS'] = '1'
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # init model
        model = Model(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).to(device))
        mse_loss = nn.MSELoss(reduction='mean')

        cnt_wait = 0
        best = 1e9
        best_t = 0
        batch_num = nb_nodes // batch_size + 1

        # Training
        for epoch in range(args.num_epoch):

            model.train()

            all_idx = list(range(nb_nodes))
            random.shuffle(all_idx)
            total_loss = 0.

            subgraphs_1 = generate_rwr_subgraph(dgl_graph, subgraph_size)
            subgraphs_2 = generate_rwr_subgraph(dgl_graph, subgraph_size)

            for batch_idx in range(batch_num):

                optimiser.zero_grad()

                is_final_batch = (batch_idx == (batch_num - 1))

                if not is_final_batch:
                    idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                else:
                    idx = all_idx[batch_idx * batch_size:]

                cur_batch_size = len(idx)

                lbl = torch.unsqueeze(torch.cat((torch.ones(cur_batch_size),
                                                 torch.zeros(cur_batch_size * args.negsamp_ratio))), 1).to(device)

                ba1 = []
                ba2 = []
                bf1 = []
                bf2 = []
                raw_bf1 = []
                raw_bf2 = []
                added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size)).to(device)
                added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1)).to(device)
                added_adj_zero_col[:, -1, :] = 1.
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size)).to(device)

                for i in idx:
                    cur_adj_1 = adj[:, subgraphs_1[i], :][:, :, subgraphs_1[i]]
                    cur_feat_1 = features[:, subgraphs_1[i], :]
                    raw_cur_feat_1 = raw_features[:, subgraphs_1[i], :]
                    cur_adj_2 = adj[:, subgraphs_2[i], :][:, :, subgraphs_2[i]]
                    cur_feat_2 = features[:, subgraphs_2[i], :]
                    raw_cur_feat_2 = raw_features[:, subgraphs_2[i], :]
                    ba1.append(cur_adj_1)
                    bf1.append(cur_feat_1)
                    raw_bf1.append(raw_cur_feat_1)
                    ba2.append(cur_adj_2)
                    bf2.append(cur_feat_2)
                    raw_bf2.append(raw_cur_feat_2)

                ba1 = torch.cat(ba1)
                ba1 = torch.cat((ba1, added_adj_zero_row), dim=1)
                ba1 = torch.cat((ba1, added_adj_zero_col), dim=2)
                ba2 = torch.cat(ba2)
                ba2 = torch.cat((ba2, added_adj_zero_row), dim=1)
                ba2 = torch.cat((ba2, added_adj_zero_col), dim=2)

                bf1 = torch.cat(bf1)
                bf1 = torch.cat((bf1[:, :-1, :], added_feat_zero_row, bf1[:, -1:, :]), dim=1)
                bf2 = torch.cat(bf2)
                bf2 = torch.cat((bf2[:, :-1, :], added_feat_zero_row, bf2[:, -1:, :]), dim=1)

                raw_bf1 = torch.cat(raw_bf1)
                raw_bf1 = torch.cat((raw_bf1[:, :-1, :], added_feat_zero_row, raw_bf1[:, -1:, :]), dim=1)
                raw_bf2 = torch.cat(raw_bf2)
                raw_bf2 = torch.cat((raw_bf2[:, :-1, :], added_feat_zero_row, raw_bf2[:, -1:, :]), dim=1)

                logits, f_1, f_2 = model(bf1, bf2, raw_bf1, raw_bf2, ba1, ba2)

                loss_all = b_xent(logits, lbl)
                loss1 = torch.mean(loss_all)
                loss2 = 0.5 * (mse_loss(f_1[:, -2, :], raw_bf1[:, -1, :]) + mse_loss(f_2[:, -2, :], raw_bf2[:, -1, :]))
                loss = args.alpha * loss1 + args.beta * loss2

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
                torch.save(model.state_dict(), 'checkpoints/exp_{}.pkl'.format(args.expid))
            else:
                cnt_wait += 1

            if cnt_wait == args.patience:
                print('Early stopping!', flush=True)
                break

            print('Epoch:{} Loss:{:.8f}'.format(epoch, mean_loss), flush=True)
            print('Epoch:{} Loss:{:.8f}'.format(epoch, mean_loss), file=output_file)

        # Testing
        print('Loading {}th epoch'.format(best_t), flush=True)
        model.load_state_dict(torch.load('checkpoints/exp_{}.pkl'.format(args.expid)))

        multi_round_ano_score = np.zeros((args.auc_test_rounds, nb_nodes))

        print('Testing AUC!', flush=True)

        for round in tqdm(range(args.auc_test_rounds),desc="testing AUC"):

            all_idx = list(range(nb_nodes))
            random.shuffle(all_idx)

            subgraphs_1 = generate_rwr_subgraph(dgl_graph, subgraph_size)
            subgraphs_2 = generate_rwr_subgraph(dgl_graph, subgraph_size)

            for batch_idx in range(batch_num):

                optimiser.zero_grad()

                is_final_batch = (batch_idx == (batch_num - 1))

                if not is_final_batch:
                    idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                else:
                    idx = all_idx[batch_idx * batch_size:]

                cur_batch_size = len(idx)

                ba1 = []
                ba2 = []
                bf1 = []
                bf2 = []
                raw_bf1 = []
                raw_bf2 = []
                added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size)).to(device)
                added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1)).to(device)
                added_adj_zero_col[:, -1, :] = 1.
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size)).to(device)

                for i in idx:
                    cur_adj_1 = adj[:, subgraphs_1[i], :][:, :, subgraphs_1[i]]
                    cur_adj_2 = adj[:, subgraphs_2[i], :][:, :, subgraphs_2[i]]
                    cur_feat_1 = features[:, subgraphs_1[i], :]
                    cur_feat_2 = features[:, subgraphs_2[i], :]
                    raw_cur_feat_1 = raw_features[:, subgraphs_1[i], :]
                    raw_cur_feat_2 = raw_features[:, subgraphs_2[i], :]
                    ba1.append(cur_adj_1)
                    bf1.append(cur_feat_1)
                    ba2.append(cur_adj_2)
                    bf2.append(cur_feat_2)
                    raw_bf1.append(raw_cur_feat_1)
                    raw_bf2.append(raw_cur_feat_2)

                ba1 = torch.cat(ba1)
                ba1 = torch.cat((ba1, added_adj_zero_row), dim=1)
                ba1 = torch.cat((ba1, added_adj_zero_col), dim=2)
                ba2 = torch.cat(ba2)
                ba2 = torch.cat((ba2, added_adj_zero_row), dim=1)
                ba2 = torch.cat((ba2, added_adj_zero_col), dim=2)

                bf1 = torch.cat(bf1)
                bf1 = torch.cat((bf1[:, :-1, :], added_feat_zero_row, bf1[:, -1:, :]), dim=1)
                bf2 = torch.cat(bf2)
                bf2 = torch.cat((bf2[:, :-1, :], added_feat_zero_row, bf2[:, -1:, :]), dim=1)

                raw_bf1 = torch.cat(raw_bf1)
                raw_bf1 = torch.cat((raw_bf1[:, :-1, :], added_feat_zero_row, raw_bf1[:, -1:, :]), dim=1)
                raw_bf2 = torch.cat(raw_bf2)
                raw_bf2 = torch.cat((raw_bf2[:, :-1, :], added_feat_zero_row, raw_bf2[:, -1:, :]), dim=1)

                with torch.no_grad():
                    logits, dist = model.inference(bf1, bf2, raw_bf1, raw_bf2, ba1, ba2)
                    logits = torch.sigmoid(torch.squeeze(logits))

                if args.alpha != 0.0 and args.beta != 0.0:
                    scaler1 = MinMaxScaler()
                    scaler2 = MinMaxScaler()
                    if args.negsamp_ratio == 1:
                        ano_score_1 = - (logits[:cur_batch_size] - logits[cur_batch_size:]).cpu().numpy()
                    else:
                        pos_ano_score = logits[:cur_batch_size]
                        neg_ano_score = logits[cur_batch_size:].view(-1, cur_batch_size).mean(dim=0)
                        ano_score_1 = - (pos_ano_score - neg_ano_score).cpu().numpy()
                    ano_score_2 = dist.cpu().numpy()
                    ano_score_1 = scaler1.fit_transform(ano_score_1.reshape(-1, 1)).reshape(-1)
                    ano_score_2 = scaler2.fit_transform(ano_score_2.reshape(-1, 1)).reshape(-1)
                    ano_score = args.alpha * ano_score_1 + args.beta * ano_score_2
                elif args.alpha != 0.0 and args.beta == 0.0:
                    if args.negsamp_ratio == 1:
                        ano_score = - (logits[:cur_batch_size] - logits[cur_batch_size:]).cpu().numpy()
                    else:
                        pos_ano_score = logits[:cur_batch_size]
                        neg_ano_score = logits[cur_batch_size:].view(-1, cur_batch_size).mean(dim=0)
                        ano_score = - (pos_ano_score - neg_ano_score).cpu().numpy()
                elif args.alpha == 0.0 and args.beta != 0.0:
                    ano_score = dist.cpu().numpy()
                else:
                    raise Exception("alpha and beta cannot be zero at the same time.")

                multi_round_ano_score[round, idx] = ano_score

        ano_score_final = np.mean(multi_round_ano_score, axis=0)
        
        # Compute the statistic for the given list
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

    


if __name__ == '__main__':
    datasets = ['cora','citeseer','pubmed','ACM','BlogCatalog','Flickr']
    subgraph_sizes = [2,3,4,5,6,7,8,9]
    alphas = [0, 0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99,1.0]
    ratios = {'cora':0.051,'citeseer':0.045,'pubmed':0.025,'ACM':0.036,'BlogCatalog':0.058,
              'Flickr':0.059}
    seed_number = 2
    for dataset in datasets:
        anomaly_scores = pd.DataFrame()

        ratio = ratios[dataset]
        output_file_0 = open(f'Output/SL-GAD-{dataset}.txt', 'w')
        first_line = 'seed,subgraph_size,alpha,AUC,CST,CST_equal,CST_base,loss_start_epoch,loss_end_epoch'
        print(first_line,file=output_file_0)
        for subgraph_size in subgraph_sizes:
            
            for alpha in alphas:
                start_time = time.time()
                print("Alg:SL-GAD")
                print(f"seed:{seed_number}")
                print(f"Dataset:{dataset}")
                print(f"subgraph_size:{subgraph_size}")
                print(f"alpha:{alpha}")
                
                args = padding_args(subgraph_size=subgraph_size, alpha=alpha,seed=seed_number)
                current_hp = f'{args.seed},{subgraph_size},{alpha},'
                output_file = open(f'Output/details/{dataset}/SL-GAD-{subgraph_size}-{alpha}-output.txt', 'w')
                AUC,cs,cse,ccsb,first_loss_value,last_loss_value,ano_score_final = train_test_model(args=args,output_file=output_file,ratio=ratio)
                results = f'{AUC},{cs},{cse},{ccsb},{first_loss_value},{last_loss_value}'
                output_line = current_hp+results
                print(output_line,file=output_file_0)
                # End the timer
                col_name = f'{args.seed},{subgraph_size},{alpha}'
                anomaly_scores[col_name] = ano_score_final
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
                print(formatted_time)
        csv_filename = f'Output/anomaly_scores_{dataset}.csv'
        anomaly_scores.to_csv(csv_filename, index=False)
        output_file_0.close()