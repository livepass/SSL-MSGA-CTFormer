import os
import sys
import torch
import torch.nn as nn
os.environ["MKL_THREADING_LAYER"] = "GNU"
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from tqdm import tqdm
from time import strftime, localtime
import copy
import pandas as pd
import pickle  # Used to load scaler

#from data_utils import restored_pm25

class Trainer:
    def __init__(self, args, logger, model, train_dataloader, test_dataloader, optimizer) -> None:
        self.args = args    
        self.model = model    
        self.train_dataloader = train_dataloader  
        self.test_dataloader = test_dataloader   
        self.logger = logger  
        self.optimizer = optimizer    
  
        # Load scaler -- 1119 added
        with open('./scaler.pkl', 'rb') as f:  # Load Scaler object from file
            self.scaler = pickle.load(f)
    
        self.logger.info('training arguments:')  
        for arg in vars(self.args):  
            self.logger.info('>>> {0}: {1}'.format(arg, getattr(self.args, arg)))  

    def train(self):  
        best_loss = np.inf  
        best_mse, best_rmse, best_mae, best_r2, best_mape = None, None, None, None, None  # 1128 Added: Added R² and MAPE
        self.criterion_mse = nn.MSELoss()   
        self.criterion_smoothl1 = nn.SmoothL1Loss() # Initialize variables, define loss function

        # Added error counter 1224 added
        consecutive_errors = 0  # Used to track consecutive error epochs
        error_threshold = 10    # Set threshold: stop training when 10 consecutive epochs error
        
        for epoch in range(self.args.epochs):  
            self.logger.info('>' * 60)  
            self.logger.info('epoch: {}'.format(epoch+1)) # Log for each epoch   
            self.model.train()  # Set model to training mode to enable batch normalization and dropout
            loss = 0   
            
            train_loss_all = [] # Initialize total loss and loss record list

            # Batch processing and backpropagation
            for index, (x, y) in enumerate(tqdm(self.train_dataloader, desc='Train')):  
                inputs = x.to(self.args.device)
                # print(inputs.shape) (batch_size,his_len,site_num,ferture)  
                outputs = self.model(inputs)  
                target = y.to(self.args.device) 
                loss_mse = self.criterion_mse(outputs, target)  
                loss_l1 = self.criterion_smoothl1(outputs, target)  
                loss = loss_mse + loss_l1  # Use two loss functions
                loss.backward()  
                self.optimizer.step()    
                self.optimizer.zero_grad()   
                train_loss_all.append(loss.item())
            
            # Calculate and print average loss for the entire training epoch
            train_loss = np.array(train_loss_all).mean()
            print("train loss: {:.2f}".format(train_loss))

            try: # 1224 Added 
            # test_loss, mse, rmse, mae = self.evaluate()  # After each training epoch, enter evaluation and output results
                test_loss, mse, rmse, mae, r2, mape = self.evaluate()  # 1128 Added: Added R² and MAPE
                consecutive_errors = 0  # If no error, reset consecutive error counter 1224 added
                if test_loss < best_loss:  
                    best_loss = test_loss  
                    best_mse = mse  
                    best_rmse = rmse  
                    best_mae = mae  
                    best_r2 = r2  # 1128 Added: Record best R²
                    best_mape = mape  # 1128 Added: Record best MAPE
                    # save best model  
                    if not os.path.exists('./best_model1'):  
                        os.mkdir('./best_model1')    
                    # model_path = './best_model/{}_predlen_{}_mse_{:.2f}_rmse_{:.2f}_mae_{:.2f}'.format(  
                    #                                                         self.args.model_name, self.args.current_pred_len, mse, rmse, mae)
                    model_path = './best_model1/HZ_api_{}_predlen_{}_mse_{:.2f}_rmse_{:.2f}_mae_{:.2f}_r2_{:.2f}_mape_{:.2f}'.format(  
                        self.args.model_name, self.args.current_pred_len, mse, rmse, mae, r2, mape)  # 1128 Added: Save new evaluation metrics  
                    self.best_model = copy.deepcopy(self.model)  
                    self.logger.info('>> saved:{}'.format(model_path))
            except Exception as e: # 1224 Added 
                self.logger.error(f"Error during evaluation: {e}")
                consecutive_errors += 1  # If error, increase consecutive error count
                if consecutive_errors >= error_threshold:  # Check if consecutive errors exceed threshold
                    self.logger.warning(f"Consecutive errors reached {error_threshold}. Stopping training for this model.")
                    break  # End training for current model, skip to next model           

        # End training. After training, record final best model and metrics  
        self.logger.info('>' * 60)  
        self.logger.info('save best model')  
        torch.save(self.best_model, model_path)    
        self.logger.info('mse: {:.2f}, rmse: {:.2f}, mae: {:.2f}, r2: {:.2f}, mape: {:.2f}'.format(best_mse, best_rmse, best_mae, best_r2, best_mape))  
    def evaluate(self):# After each training epoch, enter evaluation
        try:
            loss_all, mse, rmse, mae = [], [], [], []
            criterion_mae = nn.L1Loss()
            self.model.eval() # Initialize loss and evaluation metric lists
            predictions, targets = [], []#

            with torch.no_grad():  # Tell PyTorch not to compute gradients in the following code block, which helps reduce memory usage and speed up computation
                val_loss_all = []
                val_rmse_all = []
                val_mae_all = []
                val_mse_all = []
                for index, (x, y) in enumerate(tqdm(self.test_dataloader, desc='Test')):# Iterate test data
                    inputs = x.to(self.args.device)    
                    outputs = self.model(inputs)   # Model prediction
                    target = y.to(self.args.device)
                    # Collect batch by batch, select only the first time step data
                    predictions.append(outputs[:, 0, :].cpu().numpy())  # First time step, all stations
                    targets.append(target[:, 0, :].cpu().numpy())       # Same as above

                    loss_mse = self.criterion_mse(outputs, target)   
                    loss_rmse = torch.sqrt(loss_mse)
                    loss_mae = criterion_mae(outputs, target)
                    loss_l1 = self.criterion_smoothl1(outputs, target)   
                    loss = loss_mse + loss_l1  # Calculate loss and evaluation metrics
                    
                    val_mse_all.append(loss_mse.item())
                    val_loss_all.append(loss.item())
                    val_rmse_all.append(loss_rmse.item())
                    val_mae_all.append(loss_mae.item()) # Record loss and metrics
                
                # Convert list to NumPy array
                predictions = np.vstack(predictions)  # Shape is [num_samples, 57]
                targets = np.vstack(targets)

                # Calculate R² and MAPE
                r2 = r2_score(targets, predictions)  # Calculate R²
                mape = np.mean(np.abs((targets - predictions) / targets)) * 100  # Calculate MAPE

                self.model.train() # Return model to training mode

            loss_all = np.array(val_loss_all).mean() 
            mse = np.array(val_mse_all).mean()
            rmse = np.array(val_rmse_all).mean()
            mae = np.array(val_mae_all).mean()
            # print("test loss: {:.2f}".format(loss_all))# Calculate and print average loss and metrics
            print(f"Test Loss: {loss_all:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}, MAPE: {mape:.2f}")
            return loss_all, mse, rmse, mae, r2, mape  # Return all metrics
        except ValueError as e:
            self.logger.info(f"Error occurred: {e}")
            # If NaN error encountered, return default values (ensure 6 values returned)
            raise  # If exception, raise it, training function will capture and handle it
            # return [], float('nan'), float('nan'), float('nan'), float('nan'), float('nan')  # Or other appropriate default values