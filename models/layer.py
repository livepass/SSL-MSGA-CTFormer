import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import Sequential, Linear, Sigmoid
from torch_geometric.utils import dense_to_sparse

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                                    # Original paper mgatt - start

class linear(nn.Module): 
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
    def forward(self, x):
        return self.mlp(x)

class conv2d_(nn.Module):
    def __init__(self, input_dims, output_dims, kernel_size, stride=(1, 1),
                 padding='SAME', use_bias=True, activation=F.relu,
                 bn_decay=None):
        super(conv2d_, self).__init__()
        self.activation = activation
        if padding == 'SAME':
            self.padding_size = math.ceil(kernel_size / 2)
        else:
            self.padding_size = 0  # Modified to integer
        self.conv = nn.Conv2d(input_dims, output_dims, kernel_size, stride=stride,
                              padding=0, bias=use_bias)
        self.batch_norm = nn.BatchNorm2d(output_dims, momentum=bn_decay)
        torch.nn.init.xavier_uniform_(self.conv.weight)
        if use_bias:
            torch.nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        # Correct pad parameter to integer tuple
        if isinstance(self.padding_size, int):
            pad = (self.padding_size, self.padding_size, self.padding_size, self.padding_size)
        else:
            pad = tuple(self.padding_size)
        x = F.pad(x, pad)
        x = self.conv(x)
        x = self.batch_norm(x)
        if self.activation is not None:
            x = F.relu_(x)
        return x.permute(0, 3, 2, 1)

class FC(nn.Module):
    def __init__(self, input_dims, units, activations, bn_decay, use_bias=True):
        super(FC, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        assert type(units) == list
        self.convs = nn.ModuleList([conv2d_(
            input_dims=input_dim, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
            padding='VALID', use_bias=use_bias, activation=activation,
            bn_decay=bn_decay) for input_dim, num_unit, activation in
            zip(input_dims, units, activations)])

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x

class MultiScale_spatialAttention(nn.Module):
    def __init__(self, M, d, bn_decay):
        super(MultiScale_spatialAttention, self).__init__()
        self.d = d
        self.M = M
        D = self.M * self.d  # D=48
        self.FC_q = FC(input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC_k = FC(input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC_v = FC(input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay)

    def forward(self, X):
        # X shape: [batch, seq_len, node_num, D=48]
        query = self.FC_q(X)  # [batch, seq_len, node_num, D=48]
        key = self.FC_k(X)
        value = self.FC_v(X)

        # Split into M heads
        head_dim = self.d  # Dimension of each head
        # Reshape for multi-head attention
        # [batch, seq_len, node_num, M, d]
        query = query.view(X.size(0), X.size(1), X.size(2), self.M, self.d)
        key = key.view(X.size(0), X.size(1), X.size(2), self.M, self.d)
        value = value.view(X.size(0), X.size(1), X.size(2), self.M, self.d)

        # Permute to [batch * M, seq_len, node_num, d]
        query = query.permute(0, 3, 1, 2, 4).contiguous().view(-1, X.size(1), X.size(2), self.d)
        key = key.permute(0, 3, 1, 2, 4).contiguous().view(-1, X.size(1), X.size(2), self.d)
        value = value.permute(0, 3, 1, 2, 4).contiguous().view(-1, X.size(1), X.size(2), self.d)

        # Compute attention
        # Attention score: [batch*M, seq_len, node_num, node_num]
        attention = torch.matmul(query, key.transpose(-2, -1))  # [batch*M, seq_len, node_num, node_num]
        attention /= math.sqrt(self.d)
        attention = F.softmax(attention, dim=-1)

        # Apply attention to value
        # [batch*M, seq_len, node_num, d]
        X = torch.matmul(attention, value)  # [batch*M, seq_len, node_num, d]

        # Reshape back
        # [batch, M, seq_len, node_num, d]
        X = X.view(self.M, X.size(0) // self.M, X.size(1), X.size(2), self.d)
        # Permute to [batch, seq_len, node_num, M, d]
        X = X.permute(1, 2, 3, 0, 4).contiguous()
        # Reshape to [batch, seq_len, node_num, M*d]
        X = X.view(X.size(0), X.size(1), X.size(2), self.M * self.d)

        X = self.FC(X)  # [batch, seq_len, node_num, D=48]
        return X

class MultiScale_graphAttention(nn.Module):
    def __init__(self, M, d, bn_decay, mask=True):
        super(MultiScale_graphAttention, self).__init__()
        self.d = d
        self.M = M
        D = self.M * self.d  # D=48
        self.mask = mask
        self.FC_q = FC(input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC_k = FC(input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC_v = FC(input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay)

    def forward(self, X):
        # X shape: [batch, seq_len, node_num, D=48]
        query = self.FC_q(X)  # [batch, seq_len, node_num, D=48]
        key = self.FC_k(X)
        value = self.FC_v(X)

        # Split into M heads
        head_dim = self.d  # Dimension of each head
        # Reshape for multi-head attention
        # [batch, seq_len, node_num, M, d]
        query = query.view(X.size(0), X.size(1), X.size(2), self.M, self.d)
        key = key.view(X.size(0), X.size(1), X.size(2), self.M, self.d)
        value = value.view(X.size(0), X.size(1), X.size(2), self.M, self.d)

        # Permute to [batch * M, seq_len, node_num, d]
        query = query.permute(0, 3, 1, 2, 4).contiguous().view(-1, X.size(1), X.size(2), self.d)
        key = key.permute(0, 3, 1, 2, 4).contiguous().view(-1, X.size(1), X.size(2), self.d)
        value = value.permute(0, 3, 1, 2, 4).contiguous().view(-1, X.size(1), X.size(2), self.d)

        # Compute attention
        # Attention score: [batch*M, seq_len, node_num, node_num]
        attention = torch.matmul(query, key.transpose(-2, -1))  # [batch*M, seq_len, node_num, node_num]
        attention /= math.sqrt(self.d)
        attention = F.softmax(attention, dim=-1)

        # Apply attention to value
        # [batch*M, seq_len, node_num, d]
        X = torch.matmul(attention, value)  # [batch*M, seq_len, node_num, d]

        # Reshape back
        # [batch, M, seq_len, node_num, d]
        X = X.view(self.M, X.size(0) // self.M, X.size(1), X.size(2), self.d)
        # Permute to [batch, seq_len, node_num, M, d]
        X = X.permute(1, 2, 3, 0, 4).contiguous()
        # Reshape to [batch, seq_len, node_num, M*d]
        X = X.view(X.size(0), X.size(1), X.size(2), self.M * self.d)

        X = self.FC(X)  # [batch, seq_len, node_num, D=48]
        return X

class MultiScale_gatedFusion(nn.Module):
    def __init__(self, D, bn_decay):
        super(MultiScale_gatedFusion, self).__init__()
        self.FC_xs = FC(input_dims=D, units=D, activations=None, bn_decay=bn_decay, use_bias=False)
        self.FC_xt = FC(input_dims=D, units=D, activations=None, bn_decay=bn_decay, use_bias=True)
        self.FC_h = FC(input_dims=[D, D], units=[D, D], activations=[F.relu, None], bn_decay=bn_decay)

    def forward(self, HS, HG):
        XS = self.FC_xs(HS)  # [batch, seq_len, node_num, D=48]
        XG = self.FC_xt(HG)  # [batch, seq_len, node_num, D=48]
        z = torch.sigmoid(torch.add(XS, XG))  # [batch, seq_len, node_num, D=48]
        H = torch.add(torch.mul(z, HS), torch.mul(1 - z, HG))  # [batch, seq_len, node_num, D=48]
        H = self.FC_h(H)  # [batch, seq_len, node_num, D=48]
        return H
                    # Use gated (normalization, linear) method to fuse region-scale and city-scale graphs, where the weight for generating city-scale is fixed - start
class MultiScale_MGABlock(nn.Module):
    def __init__(self, M, d, bn_decay, mask=False):
        super(MultiScale_MGABlock, self).__init__()
        self.MultiScale_spatialAttention = MultiScale_spatialAttention(M, d, bn_decay)
        self.MultiScale_graphAttention = MultiScale_graphAttention(M, d, bn_decay, mask=mask)
        self.MultiScale_gatedFusion = MultiScale_gatedFusion(M * d, bn_decay)

    def forward(self, X):
        HS = self.MultiScale_spatialAttention(X)  # [batch, seq_len, node_num, D=48]
        HT = self.MultiScale_graphAttention(X)    # [batch, seq_len, node_num, D=48]
        # fusion gate
        H = self.MultiScale_gatedFusion(HS, HT)   # [batch, seq_len, node_num, D=48]
        return torch.add(X, H)         # [batch, seq_len, node_num, D=48]

 

                                            # Use normalization method to fuse region-scale and city-scale graphs, where the weight for generating city-scale is fixed
class MultiScaleAttention(nn.Module):
    def __init__(self, scales, M, d, bn_decay):
        super(MultiScaleAttention, self).__init__()
        self.scales = scales  # ['city', 'region']
        self.attentions = nn.ModuleDict({
            scale: MultiScale_MGABlock(M, d, bn_decay) for scale in scales
        })
        self.scale_weights = nn.Parameter(torch.ones(len(scales)))  # Initialize with all ones
        self.softmax = nn.Softmax(dim=0)  # Normalize weights
        self.input_projection = nn.Linear(15, M * d)  # 15 -> 48
        self.output_projection = nn.Sequential(
            nn.Linear(M * d * len(scales), 256),
            nn.ReLU(),
            nn.Linear(256, 15)
        )  # Add hidden layer and activation function

    def forward(self, processed_graphs, scale_names):
        fused_graphs = []
        normalized_weights = self.softmax(self.scale_weights)  # Normalize weights
        for idx, (graph, scale) in enumerate(zip(processed_graphs, scale_names)):
            graph = self.input_projection(graph)  # [batch, seq_len, node_num, D=48]
            fused = self.attentions[scale](graph)  # [batch, seq_len, node_num, D=48]
            fused_graphs.append(fused * normalized_weights[idx])
        fused = torch.cat(fused_graphs, dim=-1)  # [batch, seq_len, node_num, D * num_scales = 96]
        fused = self.output_projection(fused)   # [batch, seq_len, node_num, D=15]
        return fused  # [batch, seq_len, node_num, D=15]



class MultiScale_MGAtt(nn.Module):
    def __init__(self, graph, matrix_weight, attention, M, d, bn_decay, feature_dim):
        super(MultiScale_MGAtt, self).__init__()
        self.graph = graph  # Assign graph to self.graph
        self.M = M
        self.d = d
        self.bn_decay = bn_decay
        self.graph_num = graph.graph_num  # 2 (city, region)
        self.feature_dim = feature_dim

        # Introduce independent MultiScale_MGABlock for each scale
        self.MultiScale_MGABlocks = nn.ModuleList([
            MultiScale_MGABlock(M, d, bn_decay) for _ in range(self.graph_num)
        ])

        # Multi-scale attention module
        self.multi_scale_attn = MultiScaleAttention(scales=['city', 'region'], M=M, d=d, bn_decay=bn_decay)

        # aggregation operator
        self.X_linear = nn.Linear(feature_dim, graph.node_num)

        # Graph weight parameters (if needed)
        if self.graph_num > 1 and not attention:
            self.softmax = nn.Softmax(dim=0)
            if matrix_weight:
                self.adj_w = nn.Parameter(torch.randn(self.graph_num, graph.node_num, graph.node_num))
            else:
                self.adj_w = nn.Parameter(torch.randn(1, self.graph_num), requires_grad=True)
        else:
            self.adj_w = None

        self.used_graphs = graph.get_used_graphs_names()  # ['city', 'region']
        assert len(self.used_graphs) == self.graph_num

    def forward(self, X):
        if self.graph_num > 1:
            # Copy input features as input for each scale
            processed_graphs = [X for _ in self.used_graphs]  # ['city', 'region']
            scale_names = self.used_graphs  # ['city', 'region']

            # Use multi-scale attention module to fuse graph features
            fused_graphs = self.multi_scale_attn(processed_graphs, scale_names)  # [batch, seq_len, node_num, D=15]

            # aggregation layer: Directly flatten node_num and D
            X = fused_graphs.view(X.size(0), X.size(1), -1)  # [batch, seq_len, node_num * D = 36 * 15 = 540]

        else:
            adj_for_run = self.used_graphs[0]
            X = self.X_linear(X)  # [batch, seq_len, node_num]
            adj_for_run = adj_for_run.unsqueeze(0).unsqueeze(0)  # [1, 1, node_num, node_num]
            X = X.unsqueeze(-1) * adj_for_run  # [batch, seq_len, node_num, node_num]
            X = X.view(X.size(0), X.size(1), -1)  # [batch, seq_len, node_num * node_num]
            # print(f"Aggregated feature shape: {X.shape}")  # Debug info

        return X  # [batch, seq_len, node_num * D=540]

 