# from dgl.batched_graph import batch
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, dropout, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, du, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)


class MaxReadout(nn.Module):
    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq):
        return torch.max(seq, 1).values


class MinReadout(nn.Module):
    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, seq):
        return torch.min(seq, 1).values



    def forward(self, seq, query):
        query = query.permute(0, 2, 1)
        sim = torch.matmul(seq, query)
        sim = F.softmax(sim, dim=1)
        sim = sim.repeat(1, 1, 64)
        out = torch.mul(seq, sim)
        out = torch.sum(out, 1)
        return out


class Discriminator(nn.Module):
    def __init__(self, n_h, negsamp_round):
        super(Discriminator, self).__init__()
        
        ##utilise a Bilinear() to instantiate the discriminaton score function
        self.disc_score_function = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

        self.negsamp_round = negsamp_round

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, context_embed, target_embed):
        disc_score_vec = []
        
        ##positive pair 
        ## the discriminative score of its positive pair should be close to 1
        
        disc_score_vec.append(self.disc_score_function(target_embed, context_embed))

        ##negative pair(s)
        ## the discriminative score of its positive pair should be close to 0
        ##in this implementation, the target does not change, but the context has been modified as follows
        
        c_mi = context_embed
        for _ in range(self.negsamp_round):
            """
            this operation rearranges the rows of the tensor c_mi by moving 
            the second-to-last row to the top and removing the last row. 
            The overall effect is a shift of the tensor's rows up by one 
            position, with the last row being discarded.
            """    
                    
            c_mi = torch.cat((c_mi[-2:-1, :], c_mi[:-1, :]), 0)
            
            disc_score_vec.append(self.disc_score_function(target_embed, c_mi))
            
        logits = torch.cat(tuple(disc_score_vec))

        return logits



class Model(nn.Module):
    def __init__(self, subgraph_size, n_in, n_h, activation, negsamp_round, readout, dropout):        
        super(Model, self).__init__()
        self.read_mode = readout
        self.gcn = GCN(n_in, n_h, activation, dropout)
        self.hidden_size=128
        self.subgraph_size = subgraph_size
        
        # decode
        self.network = nn.Sequential(
            nn.Linear(n_h*(self.subgraph_size-1),self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size,self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size,n_in),
            nn.PReLU()
            )
        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()

        self.disc_function = Discriminator(n_h, negsamp_round)

    def forward(self, seq1, adj, seq2, sparse=False):
        """--seq1 is processed node feature, 
           --adj is the local view adjacency matrix,
           --seq2 is raw node feature, 
        """

        ##compute reconstructed node attributes
        ##-------------------------------------
        embedding_raw = self.gcn(seq2, adj, sparse) ##using (shared) GCN to encode raw node features

        sub_size = embedding_raw.shape[1] ##namely sub_graph_size
        batch = embedding_raw.shape[0]
        temp_aa = embedding_raw[:,:sub_size-2,:]
        latent_embedding = temp_aa.reshape(batch,-1)
        
        reconstructed_embedding = self.network(latent_embedding)  ##using (shared) MLP to decode embedded node features
        
        ##compute discrimination scores for contrastive learning
        ##-------------------------------------
        embedding_processed = self.gcn(seq1, adj, sparse) ##using GCN to encode processed node features
        context_embed = self.read(embedding_processed[:, : -1, :]) ##except the last column (sub_graph_level)
        target_embed = embedding_processed[:, -1, :] ##only the last column (target)
        
        
        ##Discriminator between target node "c" and its subgraph "h_mv"
        ##aim to maximize the agreement between the target node and its subgraph-level representations
        disc_score_vec = self.disc_function(context_embed, target_embed)

        return reconstructed_embedding, disc_score_vec


