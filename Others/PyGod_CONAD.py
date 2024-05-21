#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
We test GUIDE, CONAD, GAAN, AnomalyDAE, DOMINANT with this script
"""

# =============================================================================
# Step0: Start the timer
# =============================================================================
import time
start_time = time.time()
from pygod.metric import eval_roc_auc
import numpy as np
# =============================================================================
# Step1: set random seed for reproducible results
# =============================================================================

from torch_geometric.data import Data
import scipy.sparse as sp
import torch
import random
import os
import dgl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pygod.detector import DOMINANT, GUIDE, CONAD, AnomalyDAE, GAAN
import networkx as nx
import scipy.sparse as sp
import torch
import scipy.io as sio
import pandas as pd

seed_number = 2

## Set random seed
dgl.random.seed(seed_number)
np.random.seed(seed_number)
torch.manual_seed(seed_number)
torch.cuda.manual_seed(seed_number)
torch.cuda.manual_seed_all(seed_number)
random.seed(seed_number)
os.environ['PYTHONHASHSEED'] = str(seed_number)
os.environ['OMP_NUM_THREADS'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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

# =============================================================================
# Step2: Load mat with public train/val/test split to convert it to PyGod data obj
# =============================================================================




def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset+labels_dense.ravel()] = 1
    return labels_one_hot

def load_mat(dataset, train_rate=0.3, val_rate=0.1):
    """Load .mat dataset."""
    data = sio.loadmat("{}".format(dataset)) 
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']
    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)
    labels = np.squeeze(np.array(data['Class'],dtype=np.int64) - 1)
    num_classes = np.max(labels) + 1
    labels = dense_to_one_hot(labels,num_classes)
    ano_labels = np.squeeze(np.array(label))


    if 'str_anomaly_label' in data:
        str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
        attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
    else:
        str_ano_labels = None
        attr_ano_labels = None

    num_node = adj.shape[0]
    num_train = int(num_node * train_rate)
    num_val = int(num_node * val_rate)
    all_idx = list(range(num_node))
    random.shuffle(all_idx)
    idx_train = all_idx[ : num_train]
    idx_val = all_idx[num_train : num_train + num_val]
    idx_test = all_idx[num_train + num_val : ]

    return adj, feat, labels, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels

def train_test_model(dataset, weight, eta,ratio):
    ##Load data 'BlogCatalog'  'Flickr'  'ACM'  'cora'  'citeseer'  'pubmed'
    # adj, features, labels, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label = load_mat('/Users/zlifr/Downloads/Sub-CR/dataset/pubmed.mat')

    adj, features, labels, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label = load_mat(f'./dataset/{dataset}.mat')

    # Assuming adj, features, and labels are already loaded and are in the correct format 

    # Convert sparse matrix (adj) to COO format and then to edge indices
    print("----------Converting sparse matrix to COO format--------------")
    adj_coo = sp.coo_matrix(adj)
    edge_index = torch.tensor([adj_coo.row, adj_coo.col], dtype=torch.long)
    print("---------------Converting finished---------------")
    # Convert features and labels to PyTorch tensors
    print("-----------------Converting features------------")
    features_tensor = torch.FloatTensor(features.toarray())
    labels_tensor = torch.LongTensor(ano_label)
    s_tensor = torch.LongTensor(labels)
    print("------------------Converting finished-------------")

    def generate_bool_list(input_list, max_value):
        if not input_list:  # Check if the list is empty
            return []
        return [i in input_list for i in range(max_value)]

    # Add public split of train/val/test indices mask
    idx_train_list = idx_train
    bool_list_idx_train = generate_bool_list(idx_train_list, len(ano_label))
    train_mask_tensor = torch.BoolTensor(bool_list_idx_train)

    idx_val_list = idx_val
    bool_list_idx_val = generate_bool_list(idx_val_list, len(ano_label))
    val_mask_tensor = torch.BoolTensor(bool_list_idx_val)

    idx_test_list = idx_test
    bool_list_idx_test = generate_bool_list(idx_test_list, len(ano_label))
    test_mask_tensor = torch.BoolTensor(bool_list_idx_test)


    # Create torch_geometric.data.Data object
    data = Data(x=features_tensor, 
                edge_index=edge_index,
                y=labels_tensor, 
                train_mask = train_mask_tensor,
                val_mask = val_mask_tensor,
                test_mask = test_mask_tensor)

    data.y = data.y.bool()


    ##print information
    train_count = data.train_mask.tolist().count(True)
    val_count = data.val_mask.tolist().count(True)
    test_count = data.test_mask.tolist().count(True)

    print("train/val/test: {}/{}/{}".format(train_count,val_count,test_count))

    # -----------------------------------------------------------------------------
    # Or Load PyGod Data directly from PyGOD
    # note that their split was not used by most papers, and they generally have 10% anomalies
    # -----------------------------------------------------------------------------

    # from pygod.utils import load_data

    # data = load_data('inj_cora') ##10% anomalies
    # data.y = data.y.bool()


    # =============================================================================
    # ##Step3: Initilisation
    # =============================================================================



    # -----------------------------------------------------------------------------
    # ##Testing GUIDE and its hyper-parameters
    ## HP1: alpha (trade-off hp) from {0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99} 
    # Default: ``0.1``. for Cora
    # -----------------------------------------------------------------------------

    # detector = GUIDE(hid_a=64, hid_s=4, num_layers=4, epoch=100, alpha=weight, graphlet_size=4)


    # -----------------------------------------------------------------------------
    # ##Testing CONAD and its hyper-parameters
    # weight : from {0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99} 
    #     Weight between reconstruction of node feature and structure.
    #     Default: ``0.9``. for Cora
    # eta : from {0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99} 
    #     Weight between contrastive and reconstruction.Default: ``0.7``.
    # margin : from {0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99} 
    #     Margin in margin ranking loss. Default: ``0.5``.
    # r : from {0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99}
    #     The rate of augmented anomalies. Default: ``0.2``.
    # m : from {10, 20, 30, 40, 50, 60, 70, 80, 90} 
    #     For densely connected nodes, the number of
    #     edges to add. Default: ``50``.
    # k : from {10, 20, 30, 40, 50, 60, 70, 80, 90} 
    #     Same as ``k`` in ``pygod.generator.gen_contextual_outlier``.
    #     Default: ``50``.
    # f : from {5, 10, 15, 20, 25, 30, 35, 40, 45}
    #     For disproportionate nodes, the scale factor applied
    #     on their attribute value. Default: ``10``.
    # -----------------------------------------------------------------------------

    # detector = CONAD(hid_a=64, num_layers=4, epoch=100, weight=0.50, eta=0.5, margin=0.5, r=0.2, m=50, k=50, f=10, sigmoid_s=True)
    detector = CONAD(hid_a=64, num_layers=4, epoch=100 ,weight=weight, eta=eta, margin=0.5, r=0.2, m=50, k=50, f=10, sigmoid_s=True)
    # -----------------------------------------------------------------------------
    # ##Testing AnomalyDAE and its hyper-parameters
        # alpha : from {0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99} 
        #     Weight between reconstruction of node feature and structure.
        #     Default: ``0.7``. for Cora
        # eta : from {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} 
        #     The additional penalty for nonzero attribute. Default: ``1.``. for Cora
        # theta : from {10, 20, 30, 40, 50, 60 ,70, 80, 90, 100} 
        #     The additional penalty for nonzero structure. Default: ``1.``.
    # -----------------------------------------------------------------------------

    # detector = AnomalyDAE(hid_dim=64, num_layers=4, epoch=100, alpha=1.0, eta=1, theta=10)

    # -----------------------------------------------------------------------------
    # ##Testing GAAN and its hyper-parameters
        # weight : from {0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99} 
        #     Weight between reconstruction of node feature and structure.
        #     Default: ``0.5``.
    # -----------------------------------------------------------------------------

    # detector = GAAN(hid_dim=64, num_layers=4, epoch=100, weight=weight)

    # -----------------------------------------------------------------------------
    # ##Testing DOMINANT and its hyper-parameters
        # weight : from {0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99} 
        #     Weight between reconstruction of node feature and structure.
        #     Default: ``0.9``. for Cora
    # -----------------------------------------------------------------------------




    # =============================================================================
    # ##Step6: Evaluation
    # =============================================================================




    # Compute the statistic for the given list

    # detector = DOMINANT(hid_dim=64, num_layers=4, epoch=100, weight=weight, sigmoid_s=True, sigmoid_a=False)

    # =============================================================================
    # ##Step4: Training model
    # =============================================================================
    print("Training")
    detector.fit(data)
    first_loss_value = detector.first_loss_value
    last_loss_value = detector.last_loss_value
    print("Training finished")
    print(f"First loss value:{first_loss_value}")
    print(f"Last loss value:{last_loss_value}")
    # =============================================================================
    # ##Step5: Inference
    # =============================================================================
    pred, score, prob, conf = detector.predict(data, return_pred=True, return_score=True, return_prob=True, return_conf=True)

    print('Labels:')
    print(pred)

    print('Raw scores:')
    print(score)

    print('Probability:')
    print(prob)

    print('Confidence:')
    print(conf)
    ##Scale scores 
    my_scaler = MinMaxScaler()
    ano_score_final = my_scaler.fit_transform(score.reshape(-1, 1)).reshape(-1)


    # -----------------------------------------------------------------------------
    ##Plot anomaly scores
    # -----------------------------------------------------------------------------

    # Tensor values
    tensor_values = ano_score_final.tolist()

    # Creating a histogram plot
    plt.figure(figsize=(10, 6))
    plt.hist(tensor_values, bins='auto', color='blue', alpha=0.7)
    plt.title('Histogram of Tensor Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    # plt.show()

    # -----------------------------------------------------------------------------
            
    computed_statistic = compute_custom_statistic(ano_score_final,ratio)
    computed_statistic_equal = compute_custom_statistic_equal(ano_score_final,ratio)
    compute_custom_base = compute_custom_statistic_base(ano_score_final,ratio)
    print('\n Contrast Score Margin:{:.5f}'.format(computed_statistic))
    print('\n Contrast Score Margin Equal:{:.5f}'.format(computed_statistic_equal))
    print('\n Contrast Score Margin base:{:.5f}'.format(compute_custom_base))

    cs = '{:.5f}'.format(computed_statistic)
    cse = '{:.5f}'.format(computed_statistic_equal)
    ccsb = '{:.5f}'.format(compute_custom_base)

    auc_score = eval_roc_auc(data.y, score)
    print('AUC Score:', auc_score)
    AUC = '{:.4f}'.format(auc_score)

    return AUC,cs,cse,ccsb,first_loss_value,last_loss_value,ano_score_final
# =============================================================================
# Step7: End the timer
# ============================================================================

if __name__ == '__main__':
    # Datasets
    datasets = ['cora','citeseer','pubmed','ACM']
    # Datasets with ratios
    ratios = {'cora':0.051,'citeseer':0.045,'pubmed':0.025,'ACM':0.036,'BlogCatalog':0.058,
              'Flickr':0.059}
    weights=[0, 0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99,1.0]

    etas = [0,0.01,0.5,0.99,1.0]

    for dataset in datasets:
        anomaly_scores = pd.DataFrame()
        ratio = ratios[dataset]
        output_file_0 = open(f'Output/CONAD/CONAD-{dataset}.txt', 'w')
        first_line = 'seed,weight,eta,AUC,CST,CST_equal,CST_base,loss_start_epoch,loss_end_epoch'
        print(first_line,file=output_file_0)
        for weight in weights:
            for eta in etas:
                print("Alg:CONAD")
                print(f"seed:{seed_number}")
                print(f"dataset:{dataset}")
                print(f"weight:{weight}")
                print(f"eta:{eta}")
                start_time = time.time()
                    # Padding args
                seed = f'{seed_number},'
                current_hp = f'{weight},{eta},'

                AUC,cs,cse,ccsb,first_loss_value,last_loss_value,ano_score_final = train_test_model(dataset=dataset, weight=weight,eta=eta,ratio=ratio)

                results = f'{AUC},{cs},{cse},{ccsb},{first_loss_value},{last_loss_value}'
                output_line = seed+current_hp+results
                print(output_line,file=output_file_0)

                col_name = f'{seed_number},{weight},{eta}'
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
        csv_filename = f'Output/CONAD/anomaly_scores_{dataset}.csv'
        anomaly_scores.to_csv(csv_filename, index=False)
        output_file_0.close()


        