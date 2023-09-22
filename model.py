import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from target_search_network.encoder_residues import Encoder
from target_search_network.decoder_2D_skeletons import Decoder2D
from target_search_network.decoder_3D_skeletons import Decoder3D
import numpy as np


class LayerNorm(nn.Module):    #归一化 Xi = (Xi-μ)/σ
    def __init__(self, emb_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(emb_size))
        self.beta = nn.Parameter(torch.zeros(emb_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings_add_position(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(Embeddings_add_position, self).__init__()
        self.emb_size = emb_size
        self.word_embeddings = nn.Embedding(vocab_size, emb_size)
        self.LayerNorm = LayerNorm(emb_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_dp):
        seq_length = input_dp.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_dp.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_dp).unsqueeze(2)

        words_embeddings = self.word_embeddings(input_dp)

        pe = torch.zeros(words_embeddings.shape).cuda()
        div = torch.exp(torch.arange(0., self.emb_size, 2) * -(math.log(10000.0) / self.emb_size)).double().cuda()
        pe[..., 0::2] = torch.sin(position_ids * div)
        pe[..., 1::2] = torch.cos(position_ids * div)

        embeddings = words_embeddings + pe
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class Embeddings_no_position(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(Embeddings_no_position, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, emb_size)
        self.LayerNorm = LayerNorm(emb_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self,input_dp):
        words_embeddings = self.word_embeddings(input_dp)
        embeddings = self.LayerNorm(words_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class Drug_3D_feature(nn.Module):
    def __init__(self, emb_size, p, device):
        self.emb_size = emb_size
        self.p = p
        self.device = device
        super(Drug_3D_feature, self).__init__()

    def forward(self, input_dp, pos):
        item = []
        pos = pos * self.p

        # sum PE with token embeddings
        div = torch.exp(torch.arange(0., self.emb_size, 2) * -(math.log(10000.0) / self.emb_size)).double().to(self.device)
        for i in range(3):
            pe = torch.zeros(input_dp.shape).to(self.device)
            pe[..., 0::2] = torch.sin(pos[..., i].unsqueeze(2) * div)
            pe[..., 1::2] = torch.cos(pos[..., i].unsqueeze(2) * div)
            item.append(pe)
        input_dp = input_dp.unsqueeze(1)
        drug_3d_feature = input_dp
        for i in range(0, 3):
            channel_i = input_dp + item[i].unsqueeze(1)
            drug_3d_feature = torch.cat([drug_3d_feature, channel_i], dim=1)
        return drug_3d_feature


class GAT(nn.Module):
    def __init__(self, in_features, out_features):
        super(GAT, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def _prepare_attentional_mechanism_input(self, Wh):
        batch_size = Wh.size()[0]
        N = Wh.size()[1]
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)
        Wh_repeated_alternating = Wh.repeat_interleave(N, dim=0).reshape(batch_size, N*N, self.out_features)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)
        item = all_combinations_matrix.view(batch_size, N, N, 2 * self.out_features)
        return item

    def forward(self, atoms_vector, adjacency):
        Wh = torch.matmul(atoms_vector, self.W)
        a_input = self._prepare_attentional_mechanism_input(Wh)         # 连接操作
        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(3), 0.1)

        zero_vec = -9e15 * torch.ones_like(e)                           # 极小值
        attention = torch.where(adjacency > 0, e, zero_vec)             # 不相连的用极小值代替
        attention = F.softmax(attention, dim=2)                         # 权重系数α_ij，不想连的趋近于零
        h_prime = torch.bmm(attention, Wh)
        return F.elu(h_prime)


class MdDTI(nn.Module):
    def __init__(self, l_drugs_dict, l_proteins_dict, l_substructures_dict, args):
        super(MdDTI, self).__init__()
        self.device = args.device
        self.batch_size = args.batch_size
        self.task = args.task.upper()
        self.p = args.p
        self.drug_dim = args.drug_emb_dim
        self.target_dim = args.target_emb_dim

        # target
        self.embedding_residues = Embeddings_add_position(l_proteins_dict + 1, self.target_dim)
        self.encoder_residues = Encoder(self.target_dim, 2, 1024)

        # drug 2D substructure
        self.embedding_substructures = Embeddings_add_position(l_substructures_dict + 1, self.drug_dim)
        self.Decoder2D = Decoder2D(self.drug_dim, 2, 512)

        # drug 3D atom
        self.embedding_skeletons = Embeddings_no_position(l_drugs_dict + 1, self.drug_dim)
        self.gat1 = GAT(self.drug_dim, self.drug_dim)
        self.gat2 = GAT(self.drug_dim, self.drug_dim)
        self.drug_3d_feature = Drug_3D_feature(self.drug_dim, self.p, self.device)
        self.channel_max_pool = nn.MaxPool1d(4)
        self.channel_avg_pool = nn.AvgPool1d(4)
        self.spatial_conv2d = nn.Conv2d(2, 1, (5, 5), padding=2)
        self.bn = nn.BatchNorm2d(4, eps=1e-5, momentum=0.01, affine=True)
        self.α = nn.Parameter(torch.empty(size=(1, 1)))
        nn.init.constant_(self.α.data, 0.5)
        self.Decoder3D = Decoder3D(self.drug_dim, 2, 1024)

        self.ln = nn.LayerNorm(self.drug_dim)

        # predict
        if self.task == 'DTI':
            self.predict = nn.Sequential(
                nn.Linear(self.drug_dim + self.target_dim, 512),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(512, 32),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(32, 2)
            )
        elif self.task == 'CPA':
            self.predict = nn.Sequential(
                nn.Linear(self.drug_dim + self.target_dim, 1024),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(1024, 64),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(64, 1)
            )

    def p_value(self, x):
        if x < 0.2:
            return 0.2
        elif x > 0.8:
            return 0.8
        else:
            return x

    def extract_heavy_atoms_feature(self, atoms_feature, marks, padding_tensor):
        lens = [int(sum(mark)) for mark in marks]
        max_len = max(lens)

        heavy_atoms_f = torch.zeros(atoms_feature.shape[0], max_len, atoms_feature.shape[2])
        heavy_atoms_masks = torch.ones(atoms_feature.shape[0], max_len)  # Mark the heavy atoms and filled atoms.
        for i in range(len(lens)):
            heavy_atoms_f[i, :, :] = padding_tensor
            heavy_atoms_masks[i, lens[i]:] = 0

        for i, mark in enumerate(marks):
            indices = torch.nonzero(mark).squeeze()
            if indices.dim() == 0:   # The case of only containing one heavy atom.
                indices = indices.unsqueeze(0)
            heavy_atoms_f[i, :indices.shape[0]] = atoms_feature[i, indices]

        heavy_atoms_f = heavy_atoms_f.to(self.device)
        heavy_atoms_masks = heavy_atoms_masks.to(self.device)
        return heavy_atoms_f, heavy_atoms_masks

    def spatial_attention(self, skeletons_3d_feature):
        spatial_att_max, spatial_att_avg = None, None
        for i in range(self.batch_size):
            skeletons_feature = skeletons_3d_feature[i].permute(1, 2, 0)
            channel_max_pool = self.channel_max_pool(skeletons_feature).unsqueeze(0)
            channel_avg_pool = self.channel_avg_pool(skeletons_feature).unsqueeze(0)
            if i == 0:
                spatial_att_max = channel_max_pool
                spatial_att_avg = channel_avg_pool
            else:
                spatial_att_max = torch.cat([spatial_att_max, channel_max_pool], dim=0)
                spatial_att_avg = torch.cat([spatial_att_avg, channel_avg_pool], dim=0)
        spatial_att = self.spatial_conv2d(torch.cat([spatial_att_max, spatial_att_avg], dim=3).permute(0, 3, 1, 2))
        skeletons_3d_feature = skeletons_3d_feature * torch.sigmoid(spatial_att)
        return skeletons_3d_feature

    def consistency_loss(self, data1, data2):
        loss = 0
        for i in range(self.batch_size):
            d1 = data1[i].reshape(1, -1)
            d2 = data2[i].reshape(1, -1)
            l_upper = torch.mm(d1, d2.T)
            l_down = torch.mm(torch.sqrt_(torch.sum(d1.mul(d1), dim=1).view(torch.sum(d1.mul(d1), dim=1).shape[0], 1)),
                              torch.sqrt_(torch.sum(d2.mul(d2), dim=1).view(torch.sum(d2.mul(d2), dim=1).shape[0], 1)))
            score = torch.div(l_upper, l_down)
            loss = loss + 1 - score[0][0]
        loss = loss / self.batch_size
        return loss

    def forward(self, drug, target):
        residues, residues_masks = target['residues'], target['residues_masks']
        substructures, substructures_masks = drug['substructures'], drug['substructures_masks']
        skeletons, adjs, marks = drug['skeletons'], drug['adjs'], drug['marks']
        positions, padding_tensor_idx = drug['positions'], drug['padding_tensor_idx']

        # target
        residues_emb = self.embedding_residues(residues)
        residues_feature, K, V = self.encoder_residues(residues_emb, residues_masks)

        # drug 2d substructure
        substructures_emb = self.embedding_substructures(substructures)
        substructures_feature, substructures_att = self.Decoder2D(substructures_emb, substructures_masks, K, V, residues_masks)

        # drug 3D atom
        atoms_emb = self.embedding_skeletons(skeletons)
        padding_tensor = atoms_emb[padding_tensor_idx][-1]
        atoms_feature = self.gat1(atoms_emb, adjs)
        atoms_feature = self.gat2(atoms_feature, adjs)
        skeletons_feature, skeletons_masks = self.extract_heavy_atoms_feature(atoms_feature, marks, padding_tensor)
        skeletons_feature_3d = self.drug_3d_feature(skeletons_feature, positions)
        spatial_feature = self.spatial_attention(skeletons_feature_3d)
        skeletons_feature_3d = torch.sum(self.bn(spatial_feature), dim=1)
        skeletons_feature_3d = skeletons_feature * self.p_value(self.α) + skeletons_feature_3d * (1 - self.p_value(self.α))
        skeletons_feature_3d, skeletons_att = self.Decoder3D(skeletons_feature_3d, skeletons_masks, K, V, residues_masks)


        # predict
        target_feature = self.ln(torch.amax(residues_feature, dim=1))
        drug_feature = 0.5*(self.ln(torch.amax(substructures_feature, dim=1)) + self.ln(torch.amax(skeletons_feature_3d, dim=1)))
        drug_target = torch.cat([target_feature, drug_feature], dim=1)

        out = self.predict(drug_target)

        # consistency loss
        loss2 = self.consistency_loss(substructures_att, skeletons_att)

        # x = skeletons_att[-1][:56].clone()
        # y = substructures_att[-1][:56].clone()
        # print("atts")
        # print(list(np.array(x.cpu())))
        # print(list(np.array(y.cpu())))

        return out, loss2
