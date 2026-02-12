import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import math
import numpy as np

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=1, padding=padding, dilation=dilation))
        
        # Because padding is added to both the left and right sides of the sequence, cropping is necessary
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=1, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                 self.conv2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
    

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation // 2
            layers.append(TemporalBlock(num_inputs, num_channels[i], kernel_size, dilation, padding, dropout))
            num_inputs = num_channels[i]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

# TCAN paper implementation
#add attention and enrs
class TACN_TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, num_sub_blocks, temp_attn, nheads, en_res,
                conv, key_size, kernel_size, visual, vhdropout=0.0, dropout=0.2):
        super(TACN_TemporalConvNet, self).__init__()
        layers = []
        self.vhdrop_layer = None
        # layers.append(nn.Conv1d(emb_size*2, emb_size, 1))
        self.temp_attn = temp_attn
        # self.temp_attn_share = AttentionBlock(emb_size, key_size, emb_size)
        if vhdropout != 0.0:
            print("no vhdropout")
            # self.vhdrop_layer = VariationalHidDropout(vhdropout)
            # self.vhdrop_layer.reset_mask()
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TACN_TemporalBlock(in_channels, out_channels, kernel_size, key_size, num_sub_blocks, 
                temp_attn, nheads, en_res, conv, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, 
                vhdrop_layer=self.vhdrop_layer, visual=visual, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: [batchsize, seq_len, emb_size]
        attn_weight_list = []
        if self.temp_attn:
            out = x
            for i in range(len(self.network)):
                out, attn_weight = self.network[i](out)
                # print("the len of attn_weight", len(attn_weight))
                # if len(attn_weight) == 64:
                #     attn_weight_list.append([attn_weight[18], attn_weight[19]])
                attn_weight_list.append([attn_weight[0], attn_weight[-1]])
            return out, attn_weight_list
        else:
            return self.network(x)

class TACN_TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, key_size, num_sub_blocks, temp_attn, nheads, en_res,
                conv, stride, dilation, padding, vhdrop_layer, visual, dropout=0.2):
        super(TACN_TemporalBlock, self).__init__()
        # multi head
        self.nheads = nheads
        self.visual = visual
        self.en_res = en_res
        self.conv = conv
        self.temp_attn = temp_attn
        if self.temp_attn:
            if self.nheads > 1:
                self.attentions = [TCN_AttentionBlock(n_inputs, key_size, n_inputs) for _ in range(self.nheads)]
                for i, attention in enumerate(self.attentions):
                    self.add_module('attention_{}'.format(i), attention)
                # self.cat_attentions = AttentionBlock(n_inputs * self.nheads, n_inputs, n_inputs)
                self.linear_cat = nn.Linear(n_inputs * self.nheads, n_inputs)
            else:
                self.attention = TCN_AttentionBlock(n_inputs, key_size, n_inputs)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        if self.conv:
            self.net = self._make_layers(num_sub_blocks, n_inputs, n_outputs, kernel_size, stride, dilation,
                                        padding, vhdrop_layer, dropout)
            self.init_weights()

    def _make_layers(self, num_sub_blocks, n_inputs, n_outputs, kernel_size, stride, dilation,
                     padding, vhdrop_layer, dropout=0.2):
        layers_list = []

        if vhdrop_layer is not None:
            layers_list.append(vhdrop_layer)
        layers_list.append(
            weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)))
        layers_list.append(Chomp1d(padding))
        layers_list.append(nn.ReLU())
        layers_list.append(nn.Dropout(dropout))
        for _ in range(num_sub_blocks - 1):
            layers_list.append(
                weight_norm(
                    nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)))
            layers_list.append(Chomp1d(padding))
            layers_list.append(nn.ReLU())
            layers_list.append(nn.Dropout(dropout))

        return nn.Sequential(*layers_list)

    def init_weights(self):
        layer_idx_list = []
        for name, _ in self.net.named_parameters():
            inlayer_param_list = name.split('.')
            layer_idx_list.append(int(inlayer_param_list[0]))
        layer_idxes = list(set(layer_idx_list))
        for idx in layer_idxes:
            getattr(self.net[idx], 'weight').data.normal_(0, 0.01)

        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # x: [N, emb_size, T]
        if self.temp_attn == True:
            en_res_x = None
            if self.nheads > 1:
                # will create some bugs when nheads>1
                x_out = torch.cat([att(x) for att in self.attentions], dim=1)
                out = self.net(self.linear_cat(x_out.transpose(1, 2)).transpose(1, 2))
            else:
                # x = x if self.downsample is None else self.downsample(x)
                out_attn, attn_weight = self.attention(x)
                if self.conv:
                    out = self.net(out_attn)
                else:
                    out = out_attn
                weight_x = F.softmax(attn_weight.sum(dim=2), dim=1)
                en_res_x = weight_x.unsqueeze(2).repeat(1, 1, x.size(1)).transpose(1, 2) * x
                en_res_x = en_res_x if self.downsample is None else self.downsample(en_res_x)

            res = x if self.downsample is None else self.downsample(x)

            if self.visual:
                attn_weight_cpu = attn_weight.detach().cpu().numpy()
            else:
                attn_weight_cpu = [0] * 10
            del attn_weight

            if self.en_res:
                return self.relu(out + res + en_res_x), attn_weight_cpu
            else:
                return self.relu(out + res), attn_weight_cpu

        else:
            out = self.net(x)
            res = x if self.downsample is None else self.downsample(x)
            return self.relu(out + res)  # return: [N, emb_size, T]

class TCN_AttentionBlock(nn.Module):
    def __init__(self, in_channels, key_size, value_size):
        super(TCN_AttentionBlock, self).__init__()
        self.linear_query = nn.Linear(in_channels, key_size)
        self.linear_keys = nn.Linear(in_channels, key_size)
        self.linear_values = nn.Linear(in_channels, value_size)
        self.sqrt_key_size = math.sqrt(key_size)

    def forward(self, input):
        # input is dim (N, in_channels, T) where N is the batch_size, and T is the sequence length
        # mask = np.array([[1 if i > j else 0 for i in range(input.size(2))] for j in range(input.size(2))])
        # if input.is_cuda:
        #     mask = torch.ByteTensor(mask).cuda(input.get_device())
        # else:
        #     mask = torch.ByteTensor(mask)
        # mask = mask.bool()
        mask = np.array([[1 if i > j else 0 for i in range(input.size(2))] for j in range(input.size(2))])
        mask = torch.tensor(mask, dtype=torch.bool, device=input.device)
        input = input.permute(0, 2, 1)  # input: [N, T, inchannels]
        keys = self.linear_keys(input)  # keys: (N, T, key_size)
        query = self.linear_query(input)  # query: (N, T, key_size)
        values = self.linear_values(input)  # values: (N, T, value_size)
        temp = torch.bmm(query, torch.transpose(keys, 1, 2))  # shape: (N, T, T)
        temp.data.masked_fill_(mask, -float('inf'))

        weight_temp = F.softmax(temp / self.sqrt_key_size, dim=1)  # temp: (N, T, T), broadcasting over any slice [:, x, :], each row of the matrix
        # weight_temp_vert = F.softmax(temp / self.sqrt_key_size, dim=1) # temp: (N, T, T), broadcasting over any slice [:, x, :], each row of the matrix
        # weight_temp_hori = F.softmax(temp / self.sqrt_key_size, dim=2) # temp: (N, T, T), broadcasting over any slice [:, x, :], each row of the matrix
        # weight_temp = (weight_temp_hori + weight_temp_vert)/2
        value_attentioned = torch.bmm(weight_temp, values).permute(0, 2, 1)  # shape: (N, T, value_size)

        return value_attentioned, weight_temp  # value_attentioned: [N, in_channels, T], weight_temp: [N, T, T]



# 24/10/30--SRTCN--Try reproduction
# SRTCN_TemporalBlock
class SRTCN_TemporalBlock(nn.Module):
    def __init__(self, prev_n_outputs, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super(SRTCN_TemporalBlock, self).__init__()
        self.padding = (kernel_size - 1) * dilation // 2

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=1, padding=self.padding, dilation=dilation))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=1, padding=self.padding, dilation=dilation))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.relu1, self.dropout1,
            self.conv2, self.relu2, self.dropout2
        )

        # Downsampling layer, used to adjust the input channel number to match the output
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        # Downsampling layer, used to adjust the channel number of residual_prev
        self.residual_downsample = nn.Conv1d(prev_n_outputs, n_outputs, 1) if prev_n_outputs != n_outputs else None

        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.conv1.weight, 0, 0.01)
        nn.init.normal_(self.conv2.weight, 0, 0.01)
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, 0, 0.01)
        if self.residual_downsample is not None:
            nn.init.normal_(self.residual_downsample.weight, 0, 0.01)

    def forward(self, x, residual_prev=None):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)

        # Ensure the sequence lengths of out and res are consistent
        if out.size(2) != res.size(2):
            min_len = min(out.size(2), res.size(2))
            out = out[:, :, :min_len]
            res = res[:, :, :min_len]

        if residual_prev is not None:
            # Adjust the channel number of residual_prev
            if self.residual_downsample is not None:
                residual_prev = self.residual_downsample(residual_prev)
            # Ensure the sequence length of residual_prev is consistent with out
            if residual_prev.size(2) != out.size(2):
                residual_prev = residual_prev[:, :, :out.size(2)]
            out = out + res + residual_prev
        else:
            out = out + res

        return self.relu(out), res

# SRTCN_TemporalConvNet with corrected prev_n_outputs updating
class SRTCN_TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout):
        super(SRTCN_TemporalConvNet, self).__init__()
        self.layers = nn.ModuleList()
        num_levels = len(num_channels)
        prev_n_outputs = num_inputs  # Initial prev_n_outputs is the input channel number
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = prev_n_outputs  # Use the output channel number of the previous layer as the input channel number of the current layer
            out_channels = num_channels[i]
            self.layers.append(
                SRTCN_TemporalBlock(prev_n_outputs, in_channels, out_channels, kernel_size, dilation, dropout)
            )
            prev_n_outputs = out_channels  # Update prev_n_outputs to the output channel number of the current layer

    def forward(self, x):
        residual_prev = None
        for i, layer in enumerate(self.layers):
            x, res = layer(x, residual_prev)
            residual_prev = res
        return x

