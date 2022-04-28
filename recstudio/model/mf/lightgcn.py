from recstudio.data.dataset import MFDataset
from recstudio.model import basemodel, loss_func, scorer, module
from recstudio.ann import sampler
import torch
import torch.nn.functional as F
import numpy as np 
import scipy.sparse as sp
from recstudio.model.module import aggregator 

class LightGCNConv(torch.nn.Module):
    def __init__(self, n_layers) -> None:
        super().__init__()
        self.n_layers = n_layers

    def forward(self, norm_adj, embeddings):
        all_embeddings = [embeddings]
        ego_embeddings = embeddings 
        for i in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        return all_embeddings

class LightGCN_ItemEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.item_embeddings = None
    
    def forward(self, batch_data):
        return self.item_embeddings[batch_data]

class LightGCN_UserEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.user_embeddings = None
    
    def forward(self, batch_data):
        return self.user_embeddings[batch_data]

r"""
LightGCN
#############
    LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation (SIGIR'20)
    Reference: 
        https://dl.acm.org/doi/10.1145/3397271.3401063
"""
class LightGCN(basemodel.TwoTowerRecommender):
    r"""
    LightGCN simplifies the design of GCN to make it more concise and appropriate for recommendation. 
    LightGCN learns user and item embeddings by linearly propagating them on the user-item interaction graph,  and uses the weighted sum of the embeddings learned at all layers as the final embedding.
    """
    def __init__(self, config):
        super().__init__(config)
        self.n_layers = config['n_layers']

    def init_model(self, train_data):
        super().init_model(train_data)
        self.num_users = train_data.num_users
        self.num_items = train_data.num_items
        self.norm_adj_mat = self.get_bi_norm_adj_mat(train_data)
        self.user_emb = torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)
        self.item_emb = basemodel.Embedding(train_data.num_items, self.embed_dim, padding_idx=0)
        self.LightGCNConv = LightGCNConv(self.n_layers)

    def get_bi_norm_adj_mat(self, train_data: MFDataset):
        """
        Get a binary normlized adjacency matrix as the author did in source code.
        Get the binary normlized adjacency matrix following the formula:
        
        .. math::
            norm_adj = D^{-\frac{1}{2}} A D^{-\frac{1}{2}}
        
        Returns:
            norm_adj(tensor): the binary normlized adjacency matrix in COO format.
        """
        interaction_matrix, _ = train_data.get_graph([0], value_fields='inter')
        adj_size = train_data.num_users + train_data.num_items
        rows = np.concatenate([interaction_matrix.row, interaction_matrix.col + train_data.num_users])
        cols = np.concatenate([interaction_matrix.col + train_data.num_users, interaction_matrix.row])
        vals = np.ones(len(rows))
        adj_mat = sp.coo_matrix((vals, (rows, cols)), shape=(adj_size, adj_size))
        rowsum = np.array(adj_mat.sum(axis=-1)).flatten()
        d_inv = np.power(rowsum, -0.5)
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat_inv).tocoo()
        # norm_adj = torch.sparse_csr_tensor(norm_adj.indptr, norm_adj.indices, norm_adj.data, (adj_size, adj_size), dtype=torch.float)
        norm_adj = torch.sparse_coo_tensor(np.stack([norm_adj.row, norm_adj.col]), norm_adj.data, (adj_size, adj_size), dtype=torch.float)
        return norm_adj
        
    def config_scorer(self):
        return scorer.InnerProductScorer()

    def config_loss(self):
        return loss_func.BPRLoss()

    def build_user_encoder(self, train_data):
        return LightGCN_UserEncoder()

    def build_item_encoder(self, train_data):
        return LightGCN_ItemEncoder()

    def update_encoders(self):
        self.norm_adj_mat = self.norm_adj_mat.to(self.device)
        # [num_users + num_items, dim]
        embeddings = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        # {[num_users + num_items, dim], [num_users + num_items, dim], ..., [num_users + num_items, dim]} 
        all_embeddings = self.LightGCNConv(self.norm_adj_mat, embeddings)
        # [num_users + num_items, num_layers, dim]
        all_embeddings = torch.stack(all_embeddings, dim=-2)
        # [num_users + num_items, num_layers, dim]
        all_embeddings = torch.mean(all_embeddings, dim=-2, keepdim=False)
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
        
