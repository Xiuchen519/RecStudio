from recstudio.data.dataset import MFDataset
from recstudio.model import basemodel, loss_func, scorer, module
from recstudio.ann import sampler
import torch
import torch.nn.functional as F
import numpy as np 
import scipy.sparse as sp
from recstudio.model.module import aggregator 
from recstudio.model.module.layers import SparseDropout

class NGCFConv(torch.nn.Module):
    def __init__(self, weight_size_list, mess_dropout) -> None:
        super().__init__()
        self.weight_size_list = weight_size_list
        self.mess_dropout = mess_dropout
        self.aggregator_class = aggregator.BiAggregator
        self.aggregators = torch.nn.ModuleList()
        for i, (input_size, output_size) in enumerate(zip(self.weight_size_list[ : -1], self.weight_size_list[1 : ])):
            self.aggregators.append(self.aggregator_class(input_size, output_size, dropout=self.mess_dropout[i], act=torch.nn.LeakyReLU()))

    def forward(self, norm_adj, embeddings):
        all_embeddings = [embeddings]
        ego_embeddings = embeddings 
        for i in range(len(self.aggregators)):
            side_embeddings = torch.sparse.mm(norm_adj, ego_embeddings)
            ego_embeddings = self.aggregators[i](ego_embeddings, side_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2)
            all_embeddings.append(norm_embeddings)
        return all_embeddings

class NGCFItemEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.item_embeddings = None
    
    def forward(self, batch_data):
        return self.item_embeddings[batch_data]

class NGCFIUserEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.user_embeddings = None
    
    def forward(self, batch_data):
        return self.user_embeddings[batch_data]

r"""
NGCF
#############
    Neural Graph Collaborative Filtering (SIGIR'19)
    Reference: 
        https://dl.acm.org/doi/10.1145/3331184.3331267
"""
class NGCF(basemodel.TwoTowerRecommender):
    r"""
    NGCF is a GNN-based model, which exploits the user-item graph structure by propagating embeddings on it.
    It can model high-order connectivity expressively in user-item graph in an explict manner.
    We implement NGCF by BiAggregator.
    """
    def __init__(self, config):
        super().__init__(config)
        self.layers = config['layer_size']
        self.mess_dropout = config['mess_dropout']
        self.node_dropout = config['node_dropout']

    def init_model(self, train_data):
        super().init_model(train_data)
        self.num_users = train_data.num_users
        self.num_items = train_data.num_items
        self.norm_adj_mat = self.get_si_norm_adj_mat(train_data)
        self.user_emb = torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)
        self.item_emb = basemodel.Embedding(train_data.num_items, self.embed_dim, padding_idx=0)
        self.NGCFConv = NGCFConv(self.layers, self.mess_dropout)
        if self.node_dropout != None:
            self.sparseDropout = SparseDropout(self.node_dropout)

    def get_si_norm_adj_mat(self, train_data: MFDataset):
        """
        Get a single normlized adjacency matrix as the author did in source code.
        Get the single normlized adjacency matrix following the formula:
        
        .. math::
            norm_adj = D^{-1} A
        
        Returns:
            norm_adj(tensor): the single normlized adjacency matrix in COO format.

        """
        interaction_matrix, _ = train_data.get_graph([0], value_fields='inter')
        adj_size = train_data.num_users + train_data.num_items
        rows = np.concatenate([interaction_matrix.row, interaction_matrix.col + train_data.num_users])
        cols = np.concatenate([interaction_matrix.col + train_data.num_users, interaction_matrix.row])
        vals = np.ones(len(rows))
        adj_mat = sp.coo_matrix((vals, (rows, cols)), shape=(adj_size, adj_size))
        rowsum = np.array(adj_mat.sum(axis=-1)).flatten()
        d_inv = np.power(rowsum, -1)
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj_mat).tocoo()
        norm_adj = torch.sparse_coo_tensor(np.stack([norm_adj.row, norm_adj.col]), norm_adj.data, (adj_size, adj_size), dtype=torch.float)
        return norm_adj

    def config_scorer(self):
        return scorer.InnerProductScorer()

    def config_loss(self):
        return loss_func.BPRLoss()

    def build_user_encoder(self, train_data):
        return NGCFIUserEncoder()

    def build_item_encoder(self, train_data):
        return NGCFItemEncoder()

    def update_encoders(self):
        self.norm_adj_mat = self.norm_adj_mat.to(self.device)
        if self.node_dropout != None: 
            # node dropout
            norm_adj = self.sparseDropout(self.norm_adj_mat)
        else:
            norm_adj = self.norm_adj_mat
        # [num_users + num_items, dim]
        embeddings = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        all_embeddings = self.NGCFConv(norm_adj, embeddings)
        all_embeddings = torch.cat(all_embeddings, dim=-1)
        self.user_encoder.user_embeddings, self.item_encoder.item_embeddings = \
             torch.split(all_embeddings, [self.num_users, self.num_items], dim=0)
    
    def forward(self, batch_data, full_score):
        self.update_encoders()
        return super().forward(batch_data, full_score)

    def get_item_vector(self):
        if self.item_encoder.item_embeddings == None:
            return self.item_emb.weight[1:].detach().clone()
        else:
            return self.item_encoder.item_embeddings[1:].detach().clone()

    def prepare_testing(self):
        self.update_encoders()
        super().prepare_testing()
        
