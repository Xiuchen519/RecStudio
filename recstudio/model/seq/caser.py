from recstudio.model import basemodel, loss_func, scorer
from recstudio.data import dataset
from recstudio.ann import sampler
import torch

r"""
Caser
######################

Paper Reference:
     Jiaxi Tang, et al. "Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding" in WSDM 2018.
     https://dl.acm.org/doi/abs/10.1145/3159652.3159656
"""

class Caser(basemodel.TwoTowerRecommender):
    r"""
        | Caser models user's behavior with multi CNN layers of different kernel size, which aim
          to capture different length of behavior sequences.The idea is to embed a sequence of
          recent itemsinto an “image” in the time and latent spaces and learn sequential patterns
          as local features of the image using convolutional filters.
    """
    def init_model(self, train_data):
        super().init_model(train_data)
        self.n_v = self.config['n_v']
        self.n_h = self.config['n_h']
        self.dropout_rate = self.config['dropout_rate']
        self.max_seq_len = train_data.config['max_seq_len']
        # self.vertical_filter = torch.nn.Conv2d(in_channels=1, out_channels=self.n_v, kernel_size=(self.max_seq_len, 1))
        self.vertical_filter = torch.nn.Parameter(torch.zeros(size=(self.max_seq_len, self.n_v), dtype=torch.float32), requires_grad=True)

        heights = range(1, self.max_seq_len+1)
        self.horizontal_filter = torch.nn.ModuleList([
            torch.nn.Conv2d(in_channels=1, out_channels=self.n_h, kernel_size=(h, self.embed_dim)) for h in heights
        ])
        self.activation_phi_c = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=self.dropout_rate)

        self.fc = torch.nn.Linear(self.n_h*self.max_seq_len + self.n_v*self.embed_dim, self.embed_dim, bias=True)
        self.activation = torch.nn.ReLU()
        # self.fc2 = torch.nn.Linear(self.embed_dim*2, train_data.num_items, bias=True)
        self.item_seq_emb = torch.nn.Embedding(train_data.num_items, self.embed_dim)
        # TODO: due to the last prediction layer is a full-connected layer, we need bias here
        # use item encoder to represent the fc layer
        # self.bias = torch.nn.Parameter(torch.rand(train_data.num_items))

    def get_dataset_class(self):
        r"""SeqDataset is used for Caser."""
        return dataset.SeqDataset

    def build_item_encoder(self, train_data):
        r"""A simple item embedding is used as item encoder in Caser.
        
        The item embedding output dimension is twice as the dimension of user embedding.
        """
        return torch.nn.Embedding(train_data.num_items, self.embed_dim*2, padding_idx=0)

    def build_user_encoder(self, train_data):
        r"""A simple user embedding is used as user encoder in Caser."""
        return torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)

    def construct_query(self, batch_data):
        r"""Construct query with user embedding and sequence with CNN layers."""
        user_hist = batch_data['in_item_id']
        # seq_len = batch_data['seqlen']
        user_id = batch_data['user_id']
        P_u = self.user_encoder(user_id)    # B x D
        E_ut = self.item_seq_emb(user_hist).unsqueeze(1) # B x 1 x L x D

        # horizontal filters
        o = [None] * self.max_seq_len
        for i, conv in enumerate(self.horizontal_filter):
            if i >= user_hist.size(1):
                # pad here
                o[i] = E_ut.new_zeros(user_hist.size(0), self.n_h).detach()
            else:
                conv_out = self.activation_phi_c(conv(E_ut).squeeze(3))     # B x n_h x ?
                pool_out = torch.nn.functional.max_pool1d(conv_out, kernel_size=conv_out.size(2)).squeeze(2)    # B x n_h
                o[i] = pool_out
        o = torch.cat(o, dim=1)     # B x L*n_h

        # vertical filters
        o_tilde = (E_ut.squeeze(1).transpose(1,2) @ self.vertical_filter[:user_hist.size(1), :]).transpose(1,2)   # Bxn_vxD
        o_tilde = o_tilde.reshape(o_tilde.size(0), -1)     # B x n_v*D

        o_all = torch.cat((o, o_tilde), dim=1)  # B x (L*n_h+D*n_v)
        z = self.activation(self.fc(self.dropout(o_all)))   # B x D

        query = torch.cat((z, P_u), dim=1)  # B x 2D
        return query

    def config_loss(self):
        r"""According to the original paper, BPR loss is applied.
            Also, other loss functions like BCE loss can be used too.
        """
        return loss_func.BPRLoss()

    def config_scorer(self):
        r"""Innerproduct operation is applied to calculate scores between query and item."""
        return scorer.InnerProductScorer()

    def build_sampler(self, train_data):
        r"""Negative items are sampled uniformly."""
        return sampler.UniformSampler(train_data.num_items-1, self.score_func)
