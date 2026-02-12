import sys
import os
os.chdir('/root/data1/ts/MGATT-LSTM-main')
print("Changed working directory to:", os.getcwd())
import random

import time
from time import strftime, localtime
import argparse
import logging
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from data_utils import AirDataset, AirDatasetFusion, AirGraph
from trainer import Trainer
from models.MGAtt_LSTM import MultiScale_MGAtt_Transformer_CTMSA



t_start = time.time()

def get_logger(args):  
    logger = logging.getLogger()  
    logger.setLevel(logging.INFO)   
    logger.addHandler(logging.StreamHandler(sys.stdout))  
    if not os.path.exists(args.log_dir):  
        os.mkdir(args.log_dir, mode=0o777)   
    log_file = '{}-{}.log'.format(args.model_name, strftime("%Y-%m-%d_%H:%M:%S", localtime()))

    logger.addHandler(logging.FileHandler("%s/%s" % (args.log_dir, log_file)))  
    return logger  
   
def setup_seed(seed):    
    torch.manual_seed(seed)   
    torch.cuda.manual_seed_all(seed)   
    np.random.seed(seed)   
    random.seed(seed)  
    torch.backends.cudnn.deterministic = True  

model_classes = {
                #  'MGAtt_RNN': MGAtt_RNN,
                #  'MGAtt_LSTM': MGAtt_LSTM,
                #  'MGAtt_GRU': MGAtt_GRU,
                #  'MGAtt_TCN': MGAtt_TCN,
                #  'MGAtt_TACN': MGAtt_TACN,
                #  'MGAtt_SRTCN': MGAtt_SRTCN,
                # 'MGAtt_Transformer':MGAtt_Transformer,
                # 'MultiScale_MGAtt_LSTM':MultiScale_MGAtt_LSTM,
                # 'MultiScale_MGAtt_GRU':MultiScale_MGAtt_GRU,
                # 'MultiScale_MGAtt_TCN':MultiScale_MGAtt_TCN,
                # 'MultiScale_MGAtt_Transformer':MultiScale_MGAtt_Transformer,
                'MultiScale_MGAtt_Transformer_CTMSA':MultiScale_MGAtt_Transformer_CTMSA                     
                 }

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data', type=str, help='The dictionary where the data is stored.')
parser.add_argument('--pollution', default='pollution', type=str, help='pollution, meteorology')
parser.add_argument('--log_dir', default='./log', type=str, help='The dictionary to store log files.')
# parser.add_argument('--adj_path', default='city_distances.csv', type=str, help='distance between each two cities')
parser.add_argument('--graph_dir', default='./data/graph', type=str, help='The dictionary where the graph data is stored.')

parser.add_argument('--city_num', default=36, type=int, help='The number of cities')# Number of stations
parser.add_argument('--hist_len', default=24, type=int, help='a past period used for forecast')
parser.add_argument('--pred_len', type=int, nargs='+', default=[3,6,12,24], help='List of future prediction lengths')
# parser.add_argument('--pred_len', default=24, type=int, help='forecast how far in the futrue')
# parser.add_argument('--threshold', default=200, type=int, help='distance threshold')
parser.add_argument('--split_rate', default=0.8, type=float)

parser.add_argument('--model_name', default='MultiScale_MGAtt_LSTM', type=str, help=', '.join(model_classes))  # Select which baseline model to use
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)

# MGAtt settings
#parser.add_argument('--graph_use', default=['dist', 'neighb', 'func'], help='multi graph')
# 1/12 Added graph
# parser.add_argument('--graph_use', default=['dist', 'neighb', 'func', 'region'], help='multi graph')  # Added 'region'
# 1/13 Changed to two scales
parser.add_argument('--graph_use', default=[ 'city' , 'region'], nargs='+', help='multi graph')
parser.add_argument('--matrix_weight', default=True, help='matrix weight whether train')
parser.add_argument('--attention', default=True, help='attention')
parser.add_argument('--M', default=8, type=int, help='attention heads number')#default=8
parser.add_argument('--d', default=6, type=int, help=' dimension of Q、K、V')#default=6
parser.add_argument('--dropout', default=0.2, type=float)
parser.add_argument('--bn_decay', default=0.1, type=float)
parser.add_argument('--fix_weight', default=False, help='whether fix_weight')
parser.add_argument('--feature_dim', default=15, type=int, help='input feature dim')# Number of features

#transformer_encoder
parser.add_argument('--transformer_layers', default=12, type=int, help='number of transformer encoder layers')
parser.add_argument('--window_increment', default=2, type=int, help='increment size for window in CT-MSA')   # Modified growth rate
#parser.add_argument('--window_increment', default=3, type=int, help='increment size for window in CT-MSA')
#parser.add_argument('--window_increment', default=4, type=int, help='increment size for window in CT-MSA')
# LSTM settings
parser.add_argument('--lstm_hidden_size', default=32, type=int)
parser.add_argument('--lstm_layers', default=3, type=int)

parser.add_argument('--seed', default=10000, type=int)
parser.add_argument('--shuffle', default=True, type=bool)
parser.add_argument('--cuda', default='1', type=str, help='gpu number')
parser.add_argument('--device', default=None, type=str, help='cpu, cuda')

# GRU settings
parser.add_argument('--gru_hidden_size', default=32, type=int)
parser.add_argument('--gru_layers', default=3, type=int)

# RNN settings
parser.add_argument('--rnn_hidden_size', default=32, type=int)
parser.add_argument('--rnn_layers', default=3, type=int)

# TCN settings
parser.add_argument('--tcn_hidden_size', type=int, nargs='+', default=[32, 64, 128], help='List of hidden sizes per layer in the TCN')
parser.add_argument('--tcn_layers', type=int, default=3, help='Number of layers in the TCN')
parser.add_argument('--kernel_size', type=int, default=3, help='Size of the kernel in TCN layers')

parser.add_argument('--num_sub_blocks', type=int, default=2, help='Number of sub-blocks in each TCN block')
parser.add_argument('--temp_attn', type=bool, default=True, help='Whether to use temporal attention')
parser.add_argument('--nheads', type=int, default=1, help='Number of attention heads in TCN')
parser.add_argument('--en_res', type=bool, default=True, help='Enable residual connections')
parser.add_argument('--conv', type=bool, default=True, help='Use convolutional layers')
parser.add_argument('--visual', type=bool, default=True, help='Enable visualization of attention weights')
parser.add_argument('--key_size', type=int, default=128, help='Size of the key in attention mechanism')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device is None else torch.device(args.device)
print("choice cuda:{}".format(args.cuda))

logger = get_logger(args)

setup_seed(args.seed)

# 1119 Added, update, introduced normalization
for pred_length in args.pred_len:
    args.current_pred_len = pred_length  # Set current prediction length
    logger.info(f'\nStarting training for prediction length: {pred_length}')

    # data loader part
    logger.info('\nLoading Dataset.')
    graph = AirGraph(args)
    # Use fusion dataset (pollution + meteorology); if fusion CSV is missing, fallback internally to pollution-only
    train_dataset = AirDatasetFusion(args, mode='train')
    test_dataset = AirDatasetFusion(args, mode='test')

    # Automatically set feature_dim to match fusion feature dimension
    if hasattr(train_dataset, 'x'):
        try:
            args.feature_dim = int(train_dataset.x.shape[-1])
            logger.info(f"Auto-set feature_dim to {args.feature_dim} from fusion dataset")
        except Exception as e:
            logger.info(f"Failed to auto-set feature_dim: {e}")

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    trainset, valset = random_split(train_dataset, [train_size, val_size])

    train_dataloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(valset, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    # Train all models
    for model_name, model_cls in model_classes.items():
        args.model_name = model_name  # Update model name to current model
        logger.info(f'\nTrain model: {model_name} with pred_len {pred_length}')
        print(args.device)
        model = model_cls(args, graph).to(args.device) 
        optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        trainer = Trainer(args, logger, model, train_dataloader, test_dataloader, optimizer)
        trainer.train()

        # Print training time
        t_end = time.time()
        logger.info(f'Training {model_name} for pred_len {pred_length} took {round(t_end - t_start)} secs.')

# 1119 Commented out, update, introduced normalization
# for pred_length in args.pred_len:
#     args.current_pred_len = pred_length  # Set current prediction length
#     logger.info(f'\nStarting training for prediction length: {pred_length}')

#     # data loader part
#     logger.info('\nLoading Dataset.')
#     AirpollutDataset = AirDataset(args)
#     graph = AirGraph(args)
#     train_size = int(0.8 * len(AirpollutDataset))
#     test_size = len(AirpollutDataset) - train_size
#     trainset, testset = random_split(AirpollutDataset, [train_size, test_size])  
#     train_dataloader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=args.shuffle)
#     test_dataloader = DataLoader(dataset=testset, batch_size=args.batch_size)
    
#     # Train all models
#     for model_name, model_cls in model_classes.items():
#         args.model_name = model_name  # Update model name to current model
#         logger.info(f'\nTrain model: {model_name} with pred_len {pred_length}')
#         model = model_cls(args, graph).to(args.device) 
#         optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
#         trainer = Trainer(args, logger, model, train_dataloader, test_dataloader, optimizer)
#         trainer.train()

#         # Print training time
#         t_end = time.time()
#         logger.info(f'Training {model_name} for pred_len {pred_length} took {round(t_end - t_start)} secs.')

# # data loader part
# logger.info('\nLoading Dataset.')
# AirpollutDataset = AirDataset(args)
# graph = AirGraph(args)
# train_size = int(0.8 * len(AirpollutDataset))
# test_size = len(AirpollutDataset) - train_size
# trainset, testset = random_split(AirpollutDataset, [train_size, test_size])  
# train_dataloader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=args.shuffle)
# test_dataloader = DataLoader(dataset=testset, batch_size=args.batch_size)

# # Train all models
# for model_name, model_cls in model_classes.items():
#     args.model_name = model_name  # Update model name to current model
#     logger.info(f'\nTrain model: {model_name}')
#     model = model_cls(args, graph).to(args.device) 
#     optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
#     trainer = Trainer(args, logger, model, train_dataloader, test_dataloader, optimizer)
#     trainer.train()

#     # Print training time
#     t_end = time.time()
#     logger.info(f'Training {model_name} took {round(t_end - t_start)} secs.')