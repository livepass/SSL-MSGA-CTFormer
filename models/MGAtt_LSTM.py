import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer 
from models.layer import MultiScale_MGAtt
from models.TcnUtils import TemporalBlock
from models.TcnUtils import TACN_TemporalBlock




class MGAtt_GRU(nn.Module):
    def __init__(self, args, graph):
        super(MGAtt_GRU, self).__init__()
        self.args = args
        self.pred_len = args.current_pred_len
        # MGAtt configure
        self.MGAtt_dropout = nn.Dropout(args.dropout)
        # MGATT model
        self.MGAtt = MGAtt(graph, args.matrix_weight, args.attention, args.M, args.d, args.bn_decay, args.feature_dim)
        # GRU model
        self.encoder = nn.GRU(args.city_num * args.city_num, args.gru_hidden_size, args.gru_layers, batch_first=True, bidirectional=False)
        self.decoder = nn.GRU(args.gru_hidden_size, args.gru_hidden_size * 2, args.gru_layers, batch_first=True)
        # fully connection layer 
        self.fully_connect = nn.Linear(args.gru_hidden_size * 2, self.pred_len * args.city_num)
    
    def forward(self, input_x):
        # # -----------------------------------
        """ MGAtt Part """
        x = self.MGAtt(input_x)
        x = self.MGAtt_dropout(x) 
        # # ------------------------------------
        """ gru encoder-decoder Part """
        input_gru = x.view(x.shape[0],x.shape[1], -1) 
        # print("Converted MGAtt output shape -- GRU input shape:", input_gru.shape) # (batch_size, sequence_length, 57*57)
        encoder_seq, _ = self.encoder(input_gru)
        decoder_seq, _= self.decoder(encoder_seq)
        # # ------------------------------------

        # # ------------------------------------
        """ Fully Connect Part """
        input_tail = decoder_seq[:, -1, :]
        output = self.fully_connect(input_tail)
        fc_out = output.view(-1, self.pred_len, self.args.city_num)
        # # ------------------------------------
        return fc_out

class MGAtt_RNN(nn.Module):
    def __init__(self, args, graph):
        super(MGAtt_RNN, self).__init__()
        self.args = args
        self.pred_len = args.current_pred_len
        # MGAtt configure
        self.MGAtt_dropout = nn.Dropout(args.dropout)
        # MGATT model
        self.MGAtt = MGAtt(graph, args.matrix_weight, args.attention, args.M, args.d, args.bn_decay, args.feature_dim)
        # RNN model
        self.encoder = nn.RNN(args.city_num * args.city_num, args.rnn_hidden_size, args.rnn_layers, batch_first=True, bidirectional=False)
        self.decoder = nn.GRU(args.rnn_hidden_size, args.rnn_hidden_size * 2, args.rnn_layers, batch_first=True)
        # fully connection layer 
        self.fully_connect = nn.Linear(args.rnn_hidden_size * 2, self.pred_len * args.city_num)
    
    def forward(self, input_x):
        # # -----------------------------------
        """ MGAtt Part """
        x = self.MGAtt(input_x)
        x = self.MGAtt_dropout(x) 
        # # ------------------------------------
        """ rnn encoder-decoder Part """
        input_rnn = x.view(x.shape[0],x.shape[1], -1) 
        # print("Converted MGAtt output shape -- RNN input shape:", input_rnn.shape) # (batch_size, sequence_length, 57*57)
        encoder_seq, _ = self.encoder(input_rnn)
        decoder_seq, _= self.decoder(encoder_seq)
        # # ------------------------------------

        # # ------------------------------------
        """ Fully Connect Part """
        input_tail = decoder_seq[:, -1, :]
        output = self.fully_connect(input_tail)
        fc_out = output.view(-1, self.pred_len, self.args.city_num)
        # # ------------------------------------
        return fc_out



                        # 25/1/12 Added
class MGAtt_Transformer(nn.Module):
    def __init__(self, args, graph):
        super(MGAtt_Transformer, self).__init__()
        self.args = args
        self.pred_len = args.current_pred_len
        # MGAtt configure
        self.MGAtt_dropout = nn.Dropout(args.dropout)
        # MGATT model
        self.MGAtt = MGAtt(graph, args.matrix_weight, args.attention, args.M, args.d, args.bn_decay, args.feature_dim)
        # Transformer model
        encoder_layers = TransformerEncoderLayer(d_model=args.city_num * args.city_num, nhead=8)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=6)
        # fully connection layer 
        self.fully_connect = nn.Linear(args.city_num * args.city_num, self.pred_len * args.city_num)
    
    def forward(self, input_x):
        # MGAtt Part
        x = self.MGAtt(input_x)
        x = self.MGAtt_dropout(x) 
        # Transformer encoder
        input_transformer = x.view(x.shape[0], x.shape[1], -1).permute(1, 0, 2)  # (seq_len, batch, features)
        transformer_output = self.transformer_encoder(input_transformer)
        transformer_output = transformer_output.permute(1, 0, 2)  # (batch, seq_len, features)
        # Fully Connect Part
        input_tail = transformer_output[:, -1, :]
        output = self.fully_connect(input_tail)
        fc_out = output.view(-1, self.pred_len, self.args.city_num)
        return fc_out

                    # 1/13 Use two scales: region and city
class MultiScale_MGAtt_LSTM(nn.Module):
    def __init__(self, args, graph):
        super(MultiScale_MGAtt_LSTM, self).__init__()
        self.args = args
        self.pred_len = args.current_pred_len

        # MGAtt Configuration
        self.MultiScale_MGAtt_dropout = nn.Dropout(args.dropout)
        self.MultiScale_MGAtt = MultiScale_MGAtt(
            graph=graph,
            matrix_weight=args.matrix_weight,
            attention=args.attention,
            M=args.M,
            d=args.d,
            bn_decay=args.bn_decay,
            feature_dim=args.feature_dim
        )

        # LSTM Encoder
        # input_size is node_num * D = 36 * 15 = 540
        self.encoder = nn.LSTM(
            input_size=args.city_num * args.feature_dim, hidden_size=args.lstm_hidden_size, num_layers=args.lstm_layers,
            batch_first=True, bidirectional=False)

        # LSTM Decoder
        self.decoder = nn.LSTM(
            input_size=args.lstm_hidden_size, hidden_size=args.lstm_hidden_size * 2, num_layers=args.lstm_layers,
            batch_first=True)

        # Fully connected layer
        self.fully_connect = nn.Linear(args.lstm_hidden_size * 2, self.pred_len * args.city_num)

    def forward(self, input_x):
        # MultiScale_MGAtt part
        x = self.MultiScale_MGAtt(input_x)  # [batch, seq_len, node_num * D = 36 * 15 = 540]
        x = self.MultiScale_MGAtt_dropout(x)

        # LSTM Encoder part
        encoder_seq, _ = self.encoder(x)  # [batch, seq_len, lstm_hidden_size]

        # LSTM Decoder part
        decoder_seq, (out_h, out_c) = self.decoder(encoder_seq)  # [batch, seq_len, lstm_hidden_size * 2]

        # Fully connected part
        input_tail = decoder_seq[:, -1, :]  # [batch, lstm_hidden_size * 2]
        output = self.fully_connect(input_tail)  # [batch, pred_len * city_num]
        fc_out = output.view(-1, self.pred_len, self.args.city_num)  # [batch, pred_len, city_num]
        return fc_out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model  # Record d_model for forward check
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # [d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        if x.size(-1) != self.d_model:
            raise ValueError(f"Input feature dimension {x.size(-1)} does not match positional encoding dimension {self.d_model}")
        x = x + self.pe[:, :x.size(1), :]
        return x


class MultiScale_MGAtt_Transformer(nn.Module):
    def __init__(self, args, graph):
        super(MultiScale_MGAtt_Transformer, self).__init__()
        self.args = args
        self.pred_len = args.current_pred_len

        # MGAtt Configuration
        self.MultiScale_MGAtt_dropout = nn.Dropout(args.dropout)
        self.MultiScale_MGAtt = MultiScale_MGAtt(
            graph=graph,
            matrix_weight=args.matrix_weight,
            attention=args.attention,
            M=args.M,
            d=args.d,
            bn_decay=args.bn_decay,
            feature_dim=args.feature_dim
        )

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model=args.city_num * args.feature_dim)

        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=args.city_num * args.feature_dim, 
            nhead=6,  # d_model = args.city_num * args.feature_dim = 36 * 15 = 540 must be divisible by nhead
            dim_feedforward=4 * args.city_num * args.feature_dim,  # Usually set to 4*d_model
            dropout=args.dropout,
            activation='relu',
            batch_first=True  # Ensure input is [batch, seq_len, d_model]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=args.transformer_layers)

        # Fully connected layer
        self.fully_connect = nn.Linear(args.city_num * args.feature_dim, self.pred_len * args.city_num)

    def forward(self, input_x):
        # MultiScale_MGAtt part
        x = self.MultiScale_MGAtt(input_x)  # [batch, seq_len, node_num * D = 36 * 15 = 540]
        x = self.MultiScale_MGAtt_dropout(x)

        # Add positional encoding
        x = self.pos_encoder(x)  # [batch, seq_len, d_model]

        # Transformer Encoder part
        transformer_output = self.transformer_encoder(x)  # [batch, seq_len, d_model]

        # Fully connected part
        input_tail = transformer_output[:, -1, :]  # [batch, d_model]
        output = self.fully_connect(input_tail)  # [batch, pred_len * city_num]
        fc_out = output.view(-1, self.pred_len, self.args.city_num)  # [batch, pred_len, city_num]
        return fc_out

class CausalLocalWindowTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, window_size):
        super(CausalLocalWindowTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = F.relu
        self.window_size = window_size

    def forward(self, src):
        # src shape: [batch, seq_len, d_model]
        batch_size, seq_len, d_model = src.size()
        
        # Apply LayerNorm and Residual Connection
        src2 = self.norm1(src)
        
        # Generate causal mask for attention
        # mask shape: [seq_len, seq_len]
        # True where we want to mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=src.device), diagonal=1).bool()
        
        # Apply window size: limit attention to window_size
        if self.window_size < seq_len:
            # Create a mask where positions outside the window are masked
            window_mask = torch.zeros(seq_len, seq_len, device=src.device).bool()
            for i in range(seq_len):
                start = max(i - self.window_size + 1, 0)
                window_mask[i, :start] = True  # Mask positions before the window
            mask = mask | window_mask
        
        attn_output, attn_weights = self.self_attn(src2, src2, src2, attn_mask=mask)
        attn_output = self.dropout1(attn_output)
        src = src + attn_output
        
        # Feedforward network
        src2 = self.norm2(src)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        ff_output = self.dropout2(ff_output)
        src = src + ff_output
        
        return src

class MultiScale_MGAtt_Transformer_CTMSA(nn.Module):
    def __init__(self, args, graph):
        super(MultiScale_MGAtt_Transformer_CTMSA, self).__init__()
        self.args = args
        self.pred_len = args.current_pred_len

        # MGAtt Configuration
        self.MultiScale_MGAtt_dropout = nn.Dropout(args.dropout)
        self.MultiScale_MGAtt = MultiScale_MGAtt(
            graph=graph,
            matrix_weight=args.matrix_weight,
            attention=args.attention,
            M=args.M,
            d=args.d,
            bn_decay=args.bn_decay,
            feature_dim=args.feature_dim
        )

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model=args.city_num * args.feature_dim)

        # Transformer Encoder
        # Window size can be increased layer by layer, here set as an example
        self.transformer_layers = nn.ModuleList([
            CausalLocalWindowTransformerEncoderLayer(
                d_model=args.city_num * args.feature_dim, 
                nhead=6, 
                dim_feedforward=4 * args.city_num * args.feature_dim, 
                dropout=args.dropout,
                window_size=(i+1)*args.window_increment  # Increase window size layer by layer
            ) for i in range(args.transformer_layers)
        ])

        # Fully connected layer
        self.fully_connect = nn.Linear(args.city_num * args.feature_dim, self.pred_len * args.city_num)

    def forward(self, input_x):
        # MultiScale_MGAtt part
        x = self.MultiScale_MGAtt(input_x)  # [batch, seq_len, node_num * D = 36 * 15 = 540]
        x = self.MultiScale_MGAtt_dropout(x)
        # print(f"After MultiScale_MGAtt_dropout: {x.shape}")

        # Add positional encoding
        x = self.pos_encoder(x)  # [batch, seq_len, d_model]
        # print(f"After Positional Encoding: {x.shape}")

        # Transformer Encoder part
        for layer in self.transformer_layers:
            x = layer(x)
            # print(f"After Transformer Layer: {x.shape}")

        # Fully connected part
        input_tail = x[:, -1, :]  # [batch, d_model]
        # print(f"Input to Fully Connected: {input_tail.shape}")
        output = self.fully_connect(input_tail)  # [batch, pred_len * city_num]
        # print(f"Output before reshaping: {output.shape}")
        fc_out = output.view(-1, self.pred_len, self.args.city_num)  # [batch, pred_len, city_num]
        # print(f"Final Output: {fc_out.shape}")
        return fc_out

class MultiScale_MGAtt_GRU(nn.Module):
    def __init__(self, args, graph):
        super(MultiScale_MGAtt_GRU, self).__init__()
        self.args = args
        self.pred_len = args.current_pred_len

        # MGAtt Configuration
        self.MultiScale_MGAtt_dropout = nn.Dropout(args.dropout)
        self.MultiScale_MGAtt = MultiScale_MGAtt(
            graph=graph,
            matrix_weight=args.matrix_weight,
            attention=args.attention,
            M=args.M,
            d=args.d,
            bn_decay=args.bn_decay,
            feature_dim=args.feature_dim
        )

        # GRU Encoder
        # input_size is node_num * D = 36 * 15 = 540
        self.encoder = nn.GRU(
            input_size=args.city_num * args.feature_dim, hidden_size=args.lstm_hidden_size, num_layers=args.lstm_layers,
            batch_first=True, bidirectional=False)

        # GRU Decoder
        self.decoder = nn.GRU(
            input_size=args.lstm_hidden_size, hidden_size=args.lstm_hidden_size * 2, num_layers=args.lstm_layers,
            batch_first=True)

        # Fully connected layer
        self.fully_connect = nn.Linear(args.lstm_hidden_size * 2, self.pred_len * args.city_num)

    def forward(self, input_x):
        # MultiScale_MGAtt part
        x = self.MultiScale_MGAtt(input_x)  # [batch, seq_len, node_num * D = 36 * 15 = 540]
        x = self.MultiScale_MGAtt_dropout(x)

        # GRU Encoder part
        encoder_seq, _ = self.encoder(x)  # [batch, seq_len, gru_hidden_size]

        # GRU Decoder part
        decoder_seq, _ = self.decoder(encoder_seq)  # [batch, seq_len, gru_hidden_size * 2]

        # Fully connected part
        input_tail = decoder_seq[:, -1, :]  # [batch, gru_hidden_size * 2]
        output = self.fully_connect(input_tail)  # [batch, pred_len * city_num]
        fc_out = output.view(-1, self.pred_len, self.args.city_num)  # [batch, pred_len, city_num]
        return fc_out
class MultiScale_MGAtt_TCN(nn.Module):
    def __init__(self, args, graph):
        super(MultiScale_MGAtt_TCN, self).__init__()
        self.args = args
        self.pred_len = args.current_pred_len

        # MGAtt Configuration
        self.MultiScale_MGAtt_dropout = nn.Dropout(args.dropout)
        self.MultiScale_MGAtt = MultiScale_MGAtt(
            graph=graph,
            matrix_weight=args.matrix_weight,
            attention=args.attention,
            M=args.M,
            d=args.d,
            bn_decay=args.bn_decay,
            feature_dim=args.feature_dim
        )

        # TCN 配置
        # 修改为使用正确的输入通道数
        num_inputs = args.city_num * args.feature_dim  # 用 city_num 和 feature_dim 来定义输入通道数
        self.tcn = TemporalConvNet(num_inputs, args.tcn_hidden_size, kernel_size=args.kernel_size, dropout=args.dropout)

        # 全连接层
        self.fully_connect = nn.Linear(args.tcn_hidden_size[-1], self.pred_len * args.city_num)

    def forward(self, input_x):
        # MultiScale_MGAtt 部分
        x = self.MultiScale_MGAtt(input_x)  # [batch, seq_len, node_num * D]
        x = self.MultiScale_MGAtt_dropout(x)  # Dropout

        # 重新排列输入以适应 TCN 输入格式 (Batch, Channels, Seq_Length)
        x = x.view(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [batch, channels, seq_len]

        # TCN 部分
        tcn_output = self.tcn(x)  # [batch, channels, seq_len]

        # 使用 TCN 输出的最后一个时间步的特征
        input_tail = tcn_output[:, :, -1]  # [batch, tcn_hidden_size]

        # 全连接部分
        output = self.fully_connect(input_tail)  # [batch, pred_len * city_num]
        fc_out = output.view(-1, self.pred_len, self.args.city_num)  # [batch, pred_len, city_num]

        return fc_out
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            # Ensure num_inputs and num_channels[0] match
            layers.append(
                nn.Conv1d(num_inputs if i == 0 else num_channels[i-1], num_channels[i], kernel_size, 
                          stride=1, padding=(kernel_size-1) * dilation // 2, dilation=dilation)
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.tcn = nn.Sequential(*layers)

    def forward(self, x):
        return self.tcn(x)
