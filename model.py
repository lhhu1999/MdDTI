import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from query_network.encoder_residues import Encoder
from query_network.decoder_2D_skeletons import Decoder2D
from query_network.decoder_3D_skeletons import Decoder3D
from config import Config
import math

config = Config()


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


class Embeddings_add_sin_cos_position(nn.Module):
    def __init__(self, vocab_size, emb_size, max_size):
        super(Embeddings_add_sin_cos_position, self).__init__()
        self.emb_size = emb_size
        self.word_embeddings = nn.Embedding(vocab_size, emb_size)
        self.position_embeddings = nn.Embedding(max_size, emb_size)
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


class Drug_3d_feature(nn.Module):
    def __init__(self, emb_size):
        self.emb_size = emb_size
        super(Drug_3d_feature, self).__init__()

    def forward(self, input_dp, pos):
        item = []
        l = input_dp.shape[1]
        pos = pos * 10

        # sum PE with token embeddings
        div = torch.exp(torch.arange(0., self.emb_size, 2) * -(math.log(10000.0) / self.emb_size)).double().cuda()
        for i in range(3):
            pe = torch.zeros(input_dp.shape).cuda()
            pe[..., 0::2] = torch.sin(pos[..., i].unsqueeze(2) * div)
            pe[..., 1::2] = torch.cos(pos[..., i].unsqueeze(2) * div)
            item.append(pe)
        input_dp = input_dp.unsqueeze(1)
        drug_3d_feature = input_dp + item[0].unsqueeze(1)
        for i in range(1, 3):
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
    def __init__(self, l_drugs_dict, l_proteins_dict, l_dict):
        super(MdDTI, self).__init__()

        # drug
        self.embedding_components = Embeddings_no_position(l_dict + 1, config.atom_dim)
        self.embedding_atoms = Embeddings_no_position(l_drugs_dict + 1, config.atom_dim)
        self.gat1 = GAT(config.atom_dim, config.atom_dim)
        self.gat2 = GAT(config.atom_dim, config.atom_dim)
        self.drug_3d_feature = Drug_3d_feature(config.atom_dim)
        self.channel_max_pool = nn.MaxPool1d(3)
        self.channel_avg_pool = nn.AvgPool1d(3)
        self.spatial_conv2d = nn.Conv2d(2, 1, (5, 5), padding=2)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(3, eps=1e-5, momentum=0.01, affine=True)
        self.Decoder2D = Decoder2D(config.atom_dim, 2, 512)
        self.Decoder3D = Decoder3D(config.atom_dim, 2, 1024)

        # protein
        self.embedding_residues = Embeddings_add_sin_cos_position(l_proteins_dict + 1, config.residues_dim, config.max_length_residue)
        self.encoder_residues = Encoder(config.residues_dim, 2, 1024)

        self.ln = nn.LayerNorm(config.residues_dim)

        self.output_interaction = nn.Sequential(
            nn.Linear(config.atom_dim + config.residues_dim, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

        self.output_affinity = nn.Sequential(
            nn.Linear(config.atom_dim + config.residues_dim, 768),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(768, 24),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(24, 1)
        )

        self.α = nn.Parameter(torch.empty(size=(1, 1)))
        nn.init.constant_(self.α.data, 0.5)

    def batch_pad(self, datas, flag):
        N = max([a.shape[0] for a in datas])    # 批次里面最长的
        if flag == 0:
            if N > config.max_length_skeleton:
                N = config.max_length_skeleton
        elif flag == 1:
            if N > config.max_length_components:
                N = config.max_length_components
        else:
            if N > config.max_length_residue:
                N = config.max_length_residue

        if datas[0].ndim == 1:
            data = np.zeros((len(datas), N))
            data_mask = np.zeros((len(datas), N))
            for i, a in enumerate(datas):
                n = a.shape[0]
                if n > N:
                    n = N
                data[i, :n] = a[:n] + 1         # 前面的为a+1,后面补充的为0
                data_mask[i, :n] = 1            # 前面的为1，后面的为0
        else:
            data = np.zeros((len(datas), N, N))
            data_mask = np.zeros((len(datas), N, N))
            for i, a in enumerate(datas):
                n = a.shape[0]
                if n > N:
                    n = N
                data[i, :n, :n] = a[0:n, 0:n]
                data_mask[i, :n, :n] = 1
        return data, data_mask

    def positions_pad(self, datas):
        N = max([a.shape[0] for a in datas])  # 批次里面最长的
        if N > config.max_length_skeleton:
            N = config.max_length_skeleton
        data = np.zeros((len(datas), N, 3))
        for i, a in enumerate(datas):
            n = a.shape[0]
            if n > N:
                n = N
            data[i, :n, :3] = a[0:n, 0:3]
        return data

    def batchToTensor(self, batch_data, device):
        atoms_pad, __ = self.batch_pad(batch_data[0], 0)    # atoms_pad:前面的为a+1,后面补充的为0; atoms_mask:前面的为1，后面的为0
        adjacency_pad, _ = self.batch_pad(batch_data[1], 0)         # adjacencies_pad:小矩阵为a,后面和下面补充的为0; _:小矩阵为1，后面下面的为0
        components_pad, components_mask = self.batch_pad(batch_data[6], 1)
        residues_pad, residues_mask = self.batch_pad(batch_data[4], 2)

        atoms_pad = torch.LongTensor(atoms_pad).to(device)
        adjacency_pad = torch.LongTensor(adjacency_pad).to(device)
        positions_pad = torch.FloatTensor(self.positions_pad(batch_data[3])).to(device)
        components_pad = torch.LongTensor(components_pad).to(device)
        components_mask = torch.FloatTensor(components_mask).to(device)
        residues_pad = torch.LongTensor(residues_pad).to(device)
        residues_mask = torch.FloatTensor(residues_mask).to(device)

        return atoms_pad, adjacency_pad, positions_pad, components_pad, components_mask, residues_pad, residues_mask

    def extract_skeletons_feature(self, atoms_emb, adjacency, atoms_f, marks, device):
        init_zero_tensor = []
        for i in range(config.batch_size):
            if adjacency[i][len(adjacency[0])-1][len(adjacency[0])-1] > 0:
                init_zero_tensor.append(torch.unsqueeze(atoms_emb[i][len(adjacency[0]) - 1], dim=0))
            else:
                init_zero_tensor.append(torch.unsqueeze(atoms_f[i][len(adjacency[0]) - 1], dim=0))

        max = 0
        for e in range(config.batch_size):
            l = len(marks[e])
            if l > config.max_length_skeleton:
                l = config.max_length_skeleton
                marks[e][l:] = 0
            flag = marks[e]
            count = flag[flag==1].size
            if count == 0:
                marks[e][:] = 1
            if max < count:
                max = count
        skeletons_mask = np.zeros((config.batch_size, max))

        for e in range(config.batch_size):
            p = 0
            for j in range(len(marks[e])):
                if marks[e][j] > 0:
                    if p == 0:
                        atoms = torch.unsqueeze(atoms_f[e][j], dim=0)
                    else:
                        atoms = torch.cat((atoms, torch.unsqueeze(atoms_f[e][j], dim=0)), dim=0)
                    p = 1
            skeletons_mask[e, :len(atoms)] = 1      # 前面的为1，后面的为0
            while len(atoms) < max:
                atoms = torch.cat((atoms, init_zero_tensor[e]), dim=0)
            atoms = torch.unsqueeze(atoms, dim=0)
            if e == 0:
                skeletons_feature = atoms
            else:
                skeletons_feature = torch.cat((skeletons_feature, atoms), dim=0)
        skeletons_mask = torch.FloatTensor(skeletons_mask).to(device)
        return skeletons_feature, skeletons_mask

    def spatial_attention(self, skeletons_3d_feature):
        for i in range(config.batch_size):
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
        skeletons_3d_feature = skeletons_3d_feature * self.sigmoid(spatial_att)
        return skeletons_3d_feature

    def p_value(self, x):
        if x < 0.2:
            return 0.2
        elif x > 0.8:
            return 0.8
        else:
            return x

    def loss2(self, data1, data2):
        loss = 0
        for i in range(config.batch_size):
            d1 = data1[i].reshape(1, -1)
            d2 = data2[i].reshape(1, -1)
            l_upper = torch.mm(d1, d2.T)
            l_down = torch.mm(torch.sqrt_(torch.sum(d1.mul(d1), dim=1).view(torch.sum(d1.mul(d1), dim=1).shape[0], 1)),
                              torch.sqrt_(torch.sum(d2.mul(d2), dim=1).view(torch.sum(d2.mul(d2), dim=1).shape[0], 1)))
            score = torch.div(l_upper, l_down)
            loss = loss + 1 - score[0][0]
        loss = loss / config.batch_size
        return loss

    def forward(self, batch_data, device, task):
        atoms, adjacency, positions, components, components_mask, residues, residues_mask = self.batchToTensor(batch_data, device)

        # target
        residues_emb = self.embedding_residues(residues)
        residues_feature, K, V = self.encoder_residues(residues_emb, residues_mask)

        # drug 2d
        components_feature = self.embedding_components(components)

        # drug 3d
        atoms_emb = self.embedding_atoms(atoms)
        atoms_feature = self.gat1(atoms_emb, adjacency)
        atoms_feature = self.gat2(atoms_feature, adjacency)
        skeletons_feature, skeletons_mask = self.extract_skeletons_feature(atoms_emb, adjacency, atoms_feature, batch_data[2], device)
        skeletons_feature_3d = self.drug_3d_feature(skeletons_feature, positions)
        spatial_feature = self.spatial_attention(skeletons_feature_3d)
        skeletons_feature_3d = torch.sum(self.bn(spatial_feature), dim=1)
        skeletons_feature_3d = skeletons_feature * self.p_value(self.α) + skeletons_feature_3d * (1 - self.p_value(self.α))

        # query network
        skeletons_feature_2d, target_score_2d = self.Decoder2D(components_feature, components_mask, K, V, residues_mask)
        skeletons_feature_3d, target_score_3d = self.Decoder3D(skeletons_feature_3d, skeletons_mask, K, V, residues_mask)
        loss2 = self.loss2(target_score_2d, target_score_3d)

        protein_feature = self.ln(torch.amax(residues_feature, dim=1))
        drug_feature_2d = self.ln(torch.amax(skeletons_feature_2d, dim=1))
        drug_feature_3d = self.ln(torch.amax(skeletons_feature_3d, dim=1))

        drug_feature = drug_feature_2d + drug_feature_3d

        df_pf = torch.cat([protein_feature, drug_feature], dim=1)

        if task == 'interaction':
            out = self.output_interaction(df_pf)
        else:
            out = self.output_affinity(df_pf)

        return out, loss2
