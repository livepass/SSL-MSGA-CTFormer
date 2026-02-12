import pickle  
import os
import numpy as np
import scipy.sparse as sp 
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

class AirGraph():
    def __init__(self, args):
        graph_dir = args.graph_dir
        # self.A_dist = torch.from_numpy(np.float32(np.load(os.path.join(graph_dir, 'ChongQing_AllSite_distances.npy'))))
        # self.A_neighb = torch.from_numpy(np.float32(np.load(os.path.join(graph_dir, 'ChongQing_AllSite_neighbor_30km.npy'))))
        # self.A_func = torch.from_numpy(np.float32(np.load(os.path.join(graph_dir, 'ChongQing_AllSite_FunctionalityTest.npy'))))
        # # Added district-level graph
        # self.A_region = torch.from_numpy(np.float32(np.load(os.path.join(graph_dir, 'ChongQing_AllSite_district_graph.npy'))))

        # self.A_dist = torch.from_numpy(np.float32(np.load(os.path.join(graph_dir, 'HangZhou_AllSite_distances.npy'))))
        # self.A_neighb = torch.from_numpy(np.float32(np.load(os.path.join(graph_dir, 'HangZhou_AllSite_neighbor_30km.npy'))))
        # self.A_func = torch.from_numpy(np.float32(np.load(os.path.join(graph_dir, 'HangZhou_AllSite_FunctionalityTest.npy'))))
        # # Added district-level graph
        # self.A_region = torch.from_numpy(np.float32(np.load(os.path.join(graph_dir, 'HangZhou_AllSite_district_graph.npy'))))


        self.A_dist = torch.from_numpy(np.float32(np.load(os.path.join(graph_dir, 'BeiJing_AllSite_distances.npy'))))
        self.A_neighb = torch.from_numpy(np.float32(np.load(os.path.join(graph_dir, 'BeiJing_AllSite_neighbor_30km.npy'))))
        self.A_func = torch.from_numpy(np.float32(np.load(os.path.join(graph_dir, 'BeiJing_AllSite_FunctionalityTest.npy'))))
        # Added district-level graph. Modify k value, replace. If removed, it equals spatial interpolation method, can perform comparative ablation experiments.
        self.A_region = torch.from_numpy(np.float32(np.load(os.path.join(graph_dir, 'BeiJing_AllSite_district_graph_k2.npy'))))
        # self.A_region = torch.from_numpy(np.float32(np.load(os.path.join(graph_dir, 'BeiJing_AllSite_district_graph_k4.npy'))))



        # Merge city-related graphs
        # self.A_city = (self.A_dist + self.A_neighb + self.A_func) / 3  # Take the average
        self.A_city = (self.A_dist * 0.3591 + self.A_neighb * 0.3808 +  self.A_func * 0.2599) 
        
        # Update use_graph to ['city', 'region']
        self.use_graph = ['city', 'region']
        self.graph_num = len(self.use_graph)  # Number of graphs used: 2
        self.node_num = self.A_dist.shape[0]
        
        self.fix_weight = args.fix_weight
        if self.fix_weight:
            self.fix_weight = self.get_fix_weight()
    
    def get_used_graphs(self):
        graph_list = []
        for name in self.use_graph:
            graph_list.append(self.get_graph(name))
        return graph_list
    
    def get_used_graphs_names(self):
        return self.use_graph  # Return ['city', 'region']
    
    # fix_weight processing, default not enabled
    def get_fix_weight(self):
        return (self.A_dist * 0.3591 + \
                self.A_neighb * 0.3808 + \
                self.A_func * 0.2599) / 3
    
    def get_graph(self, name):
        if name == 'city':
            return self.A_city
        elif name == 'region':  # Added district-level graph
            return self.A_region
        else:
            raise NotImplementedError(f"Unknown graph type: {name}")


              

class AirDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.hist_len = args.hist_len
        self.pred_len = args.current_pred_len
        self.mode = mode  # train, val, or test
        self.scaler = StandardScaler()
        self.x, self.y = self._process_data(args)
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def _process_data(self, args):
        # path = './data/pollution/ChongQing_ALLSite_poll + mete_21+22.csv'
        path = './data/pollution/BeiJing_ALLSite_poll + mete_22+23.csv'
        # path = './data/pollution/HangZhou_ALLSite_poll + mete_23+24.csv'
        Airpollution = pd.read_csv(path)

        # Explicitly specify feature columns to normalize
        # features_to_normalize = [
        #     "CO", "CO_24h", "NO2", "NO2_24h", "O3", "O3_24h", "O3_8h", "O3_8h_24h",
        #     "PM10", "PM10_24h", "PM2.5", "PM2.5_24h", "SO2", "SO2_24h"
        # ]

        # features_to_normalize = [
        #     "CO", "CO_24h", "NO2", "NO2_24h", "O3", "O3_24h", "O3_8h", "O3_8h_24h",
        #     "PM10", "PM10_24h", "PM2.5", "PM2.5_24h", "SO2", "SO2_24h"
        # ]
        features_to_normalize = [
            "CO", "CO_24h", "NO2", "NO2_24h", "O3", "O3_24h", "O3_8h", "O3_8h_24h",
            "PM10", "PM10_24h", "PM2.5","PM2.5_24h", "SO2", "SO2_24h",
            "Air_Temperature", "Dew_Point_Temperature", "Sea_Level_Pressure", "Wind_Direction",
            "Wind_Speed_Rate", "Sky_Condition", "Liquid_Precip_1hr", "Liquid_Precip_6hr"
        ]
        target_column = ["AQI"] 

        # Separate processing of features to normalize and target columns
        features_data = Airpollution.loc[:, features_to_normalize]
        target_data = Airpollution.loc[:, target_column]

        # Normalize feature columns
        if self.mode == 'train':
            features_data.iloc[:, :] = self.scaler.fit_transform(features_data)
            with open('./scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
        else:
            with open('./scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            features_data.iloc[:, :] = self.scaler.transform(features_data)

        # Merge normalized features and original target columns
        Airpollution_normalized = pd.concat([features_data, target_data], axis=1)

        x_len = Airpollution_normalized.shape[0]
        all_timestamp = x_len // 36
        node_num = 36

        sample_x = []
        sample_y = []

        for i in range(self.hist_len, all_timestamp - self.pred_len):
            x = np.float32(Airpollution_normalized.iloc[node_num * (i - self.hist_len):node_num * i, :].values)
            y = np.float32(Airpollution_normalized.iloc[node_num * i:node_num * (i + self.pred_len), -1].values)
            # Reshape to [hist_len, node_num, feature_dim=15]
            x = torch.from_numpy(x).view(self.hist_len, node_num, -1)
            # Reshape y to [pred_len, node_num]
            y = torch.from_numpy(y).view(self.pred_len, node_num)
            sample_x.append(x)
            sample_y.append(y)

        sample_x = torch.stack(sample_x)  # [num_samples, hist_len, node_num, feature_dim]
        sample_y = torch.stack(sample_y)  # [num_samples, pred_len, node_num]

        return sample_x, sample_y

        
# Fusion dataset: Pollution + Meteorology (Prioritize using fusion CSV, fallback to pollution-only if missing)
class AirDatasetFusion(Dataset):
    def __init__(self, args, mode='train'):
        self.hist_len = args.hist_len
        self.pred_len = args.current_pred_len
        self.mode = mode  # train, val, or test
        self.scaler = StandardScaler()
        self.x, self.y = self._process_data(args)
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def _process_data(self, args):
        # 1) Select fusion CSV (if exists), otherwise fallback to pollution-only
        fusion_path = './data/pollution/BeiJing_ALLSite_poll + mete_22+23.csv'
        fallback_path = './data/pollution/BeiJing_ALLSite_pollution_22+23.csv'
        path = fusion_path if os.path.exists(fusion_path) else fallback_path
        Airpollution = pd.read_csv(path)

        # 2) Explicitly specify feature columns to normalize (excluding target AQI)
        features_to_normalize = [
            # Pollution (14)
            "CO", "CO_24h", "NO2", "NO2_24h", "O3", "O3_24h", "O3_8h", "O3_8h_24h",
            "PM10", "PM10_24h", "PM2.5", "PM2.5_24h", "SO2", "SO2_24h",
            # Meteorology (8)
            "Air_Temperature", "Dew_Point_Temperature", "Sea_Level_Pressure", "Wind_Direction",
            "Wind_Speed_Rate", "Sky_Condition", "Liquid_Precip_1hr", "Liquid_Precip_6hr"
        ]
        target_column = ["AQI"]

        # 3) Separate processing of features to normalize and target columns
        # If fusion file lacks meteorology columns, process only pollution columns
        available_cols = [c for c in features_to_normalize if c in Airpollution.columns]
        features_data = Airpollution.loc[:, available_cols]
        target_data = Airpollution.loc[:, target_column]

        # 4) Normalize feature columns (Save/Load independent fusion scaler)
        scaler_path = './scaler_fusion.pkl'
        if self.mode == 'train':
            features_data.iloc[:, :] = self.scaler.fit_transform(features_data)
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
        else:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            features_data.iloc[:, :] = self.scaler.transform(features_data)

        # 5) Merge normalized features and original target columns
        Airpollution_normalized = pd.concat([features_data, target_data], axis=1)

        # 6) Automatically infer station count and time steps (Prioritize via station_id, otherwise fallback to default city_num)
        if 'station_id' in Airpollution.columns:
            node_num = int(Airpollution['station_id'].nunique())
        else:
            node_num = getattr(args, 'city_num', 36)
        x_len = Airpollution_normalized.shape[0]
        all_timestamp = x_len // node_num

        # 7) Construct sample slices
        sample_x = []
        sample_y = []
        feat_dim = features_data.shape[1]

        for i in range(self.hist_len, all_timestamp - self.pred_len):
            x = np.float32(Airpollution_normalized.iloc[node_num * (i - self.hist_len):node_num * i, :feat_dim].values)
            y = np.float32(Airpollution_normalized.iloc[node_num * i:node_num * (i + self.pred_len), -1].values)
            # [hist_len, node_num, feature_dim]
            x = torch.from_numpy(x).view(self.hist_len, node_num, feat_dim)
            # [pred_len, node_num]
            y = torch.from_numpy(y).view(self.pred_len, node_num)
            sample_x.append(x)
            sample_y.append(y)

        sample_x = torch.stack(sample_x)
        sample_y = torch.stack(sample_y)

        return sample_x, sample_y


    