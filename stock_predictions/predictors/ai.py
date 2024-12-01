import yfinance as yf 
import duckdb
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from torch.nn import Module
from datetime import datetime

import os
import mlflow
import mlflow.pytorch

class StockModel(Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

class StockPrediction():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        print(f'Torch running in device: {self.device}')

    def load_yfinance_data(self, enterprise, start_date, end_date):
        yf_df = yf.download(enterprise, start=start_date, end=end_date).reset_index()

        ds = duckdb.sql(f"""
            DROP TABLE IF EXISTS finance;
            CREATE TABLE finance AS ( 
                SELECT
                "('Date', '')" as dt_time
                ,date_trunc('day', "('Date', '')") as dt
                ,"('Adj Close', '{enterprise}')" as adj_close
                ,"('Close', '{enterprise}')" as close
                ,"('High', '{enterprise}')" as high
                ,"('Low', '{enterprise}')" as low
                ,"('Open', '{enterprise}')" as open
                ,"('Volume', '{enterprise}')" as volume

                FROM yf_df
            )
            """)
        
    def get_yfinance_dataset(self):
        finance_dataset = duckdb.sql("""
            SELECT         
                * 
            FROM finance
                """)    
        

        #finance_dataset.show()
        
        return finance_dataset.df()
    
    def get_train_n_test(self, test_size = 0.65):
        yfinance_dataset = self.get_yfinance_dataset()

        timeseries = yfinance_dataset[["adj_close"]].values.astype('float32')
        dt_time = yfinance_dataset['dt']

        train_size = int(len(timeseries) * test_size)
        test_size = len(timeseries) - train_size
        train, test = timeseries[:train_size], timeseries[train_size:]
        dt_train, dt_test = dt_time[:train_size], dt_time[train_size:]

        return dt_train, train, dt_test, test
    
    def plot_yfinance_dataset(self, plot_test=False):    
        dt_train, train, dt_test, test = self.get_train_n_test()

        
        if (plot_test):
            dt_time = dt_test
            timeseries = test
        else:
            dt_time = dt_train
            timeseries = train            


        plt.plot(dt_time, timeseries)
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))

        plt.xticks(rotation=45)
        plt.show()         


    def create_torch_dataset(self, dataset, lookback):
        """Transform a time series into a prediction dataset
        
        Args:
            dataset: A numpy array of time series, first dimension is the time steps
            lookback: Size of window for prediction
        """
        X, y = [], []
        for i in range(len(dataset)-lookback):
            feature = dataset[i:i+lookback]
            target = dataset[i+1:i+lookback+1]
            X.append(feature)
            y.append(target)
        return torch.tensor(X).to(self.device), torch.tensor(y).to(self.device)
    
    def get_torch_train_n_test(self, test_percentage = 0.65, lookback = 4):
        _x_train, _y_train, _x_test, _y_test = self.get_train_n_test(0.65)
        
        x_train, y_train = self.create_torch_dataset(_y_train, lookback=lookback)
        x_test, y_test = self.create_torch_dataset(_y_test, lookback=lookback) 

        return x_train, y_train, x_test, y_test
    
    def train_model(self, model: Module, x_train, y_train, x_test, y_test, n_epochs=100, weight_decay=1e-4, log_frequency=10, prefix="prf"):
            self.model = model.to(self.device)

            optimizer = optim.Adam(self.model.parameters(), weight_decay=weight_decay)
            loss_fn = nn.MSELoss()
            loader = data.DataLoader(data.TensorDataset(x_train, y_train), shuffle=True, batch_size=64)

            run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # Start an MLflow run
            with mlflow.start_run(run_name=f"StockPrediction_{prefix}_{run_timestamp}", nested=False):
                # Log model parameters
                mlflow.log_param("n_epochs", n_epochs)
                mlflow.log_param("weight_decay", weight_decay)
                
                model_config = {
                    "input_size": 1,
                    "hidden_size": model.lstm.hidden_size,
                    "num_layers": model.lstm.num_layers,
                    "batch_first": model.lstm.batch_first,
                    "dropout": model.lstm.dropout if hasattr(model.lstm, "dropout") else 0,
                    "linear_output_size": model.linear.out_features,
                    "model_activator": model.activation
                }

                mlflow.log_params(model_config)

                for epoch in range(n_epochs):
                    self.model.train()
                    for X_batch, y_batch in loader:
                        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                        y_pred = self.model(X_batch)
                        loss = loss_fn(y_pred, y_batch)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    # Validation
                    if (epoch % log_frequency == 0) or (epoch == n_epochs - 1):  # Log metrics every 10 epochs
                        self.model.eval()
                        with torch.no_grad():
                            y_pred_train = self.model(x_train)
                            y_pred_test = self.model(x_test)

                            # Calculate metrics
                            train_loss = loss_fn(y_pred_train, y_train)
                            train_rmse = torch.sqrt(train_loss).item()
                            train_mae = torch.mean(torch.abs(y_train - y_pred_train)).item()
                            train_mape = (torch.mean(torch.abs((y_train - y_pred_train) / y_train)) * 100).item()

                            test_loss = loss_fn(y_pred_test, y_test)
                            test_rmse = torch.sqrt(test_loss).item()
                            test_mae = torch.mean(torch.abs(y_test - y_pred_test)).item()
                            test_mape = (torch.mean(torch.abs((y_test - y_pred_test) / y_test)) * 100).item()

                        # Log metrics
                        mlflow.log_metric("train_rmse", train_rmse, step=epoch)
                        mlflow.log_metric("train_mae", train_mae, step=epoch)
                        mlflow.log_metric("train_mape", train_mape, step=epoch)
                        mlflow.log_metric("test_rmse", test_rmse, step=epoch)
                        mlflow.log_metric("test_mae", test_mae, step=epoch)
                        mlflow.log_metric("test_mape", test_mape, step=epoch)

                        # Print metrics
                        print(f"Epoch {epoch}:")
                        print(f"  Train -> MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, MAPE: {train_mape:.2f}%")
                        print(f"  Test  -> MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, MAPE: {test_mape:.2f}%")

                # Log the model at the end of training
                mlflow.pytorch.log_model(self.model, "model")

    def test_model(self, model: Module, x_train, x_test, lookback=4):            
        yfinance_dataset = self.get_yfinance_dataset()
        test_size = 0.65

        timeseries = yfinance_dataset[["adj_close"]].values.astype('float32')
        train_size = int(len(timeseries) * test_size)

        with torch.no_grad():
            train_plot = np.ones((len(timeseries), 1)) * np.nan
            test_plot = np.ones((len(timeseries), 1)) * np.nan

            # Move tensors to CPU before converting to NumPy
            y_pred_train = model(x_train).cpu().numpy()[:, -1, :]
            y_pred_test = model(x_test).cpu().numpy()[:, -1, :]
            
            train_plot[lookback:lookback + len(y_pred_train), 0] = y_pred_train[:, 0]
            test_plot[train_size + lookback:train_size + lookback + len(y_pred_test), 0] = y_pred_test[:, 0]

            plt.plot(timeseries, label="Actual")
            plt.plot(train_plot, c='r', label="Train Predictions")
            plt.plot(test_plot, c='g', label="Test Predictions")
            plt.legend()
            plt.savefig("predictions_plot.png")  # Save the plot

            # Log plot as an artifact
            mlflow.log_artifact("predictions_plot.png")
            plt.show()

    def load_model_from_mlflow(self, model_uri):
        """
        Load a model from MLflow.

        Args:
            model_uri (str): The URI of the model in MLflow.
        """
        print(f"Loading model from MLflow at {model_uri}...")
        self.model = mlflow.pytorch.load_model(model_uri).to(self.device)
        print("Model loaded successfully.")

    def execute_model(self, input_data):
        """
        Execute the loaded model on new data.

        Args:
            input_data (torch.Tensor or numpy.ndarray): Input data for the model.

        Returns:
            torch.Tensor: Model predictions.
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first using 'load_model_from_mlflow'.")

        # Convert input data to a torch.Tensor if it is a numpy array
        if isinstance(input_data, np.ndarray):
            input_data = torch.tensor(input_data, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(input_data)
        return predictions
    
    def predict_and_plot(self, input_data, correct_values):
        """
        Predict with the loaded model, plot predictions against correct values, and calculate metrics.

        Args:
            input_data (torch.Tensor): Input values for the model.
            correct_values (torch.Tensor): Correct (ground truth) values to compare against predictions.

        Returns:
            dict: A dictionary containing calculated metrics.
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first using 'load_model_from_mlflow'.")

        self.model.eval()
        loss_fn = nn.MSELoss()

        with torch.no_grad():
            # Use execute_model for predictions
            predictions = self.execute_model(input_data).squeeze()
            correct_values = correct_values.squeeze()

            # Take only the last value of each sequence for comparison
            predictions = predictions[:, -1]  # Last column of predictions
            correct_values = correct_values[:, -1]  # Last column of correct values

            # Debug shapes and sample values
            print("Debugging Shapes:")
            print(f"Predictions shape: {predictions.shape}")
            print(f"Correct values shape: {correct_values.shape}")
            print("Sample Predictions:", predictions[:5])
            print("Sample Correct Values:", correct_values[:5])

            # Ensure predictions and correct values match in shape
            assert predictions.shape == correct_values.shape, (
                f"Mismatch between predictions and correct values. "
                f"Predictions: {predictions.shape}, Correct: {correct_values.shape}"
            )

            # Convert tensors to numpy arrays for plotting
            predictions = predictions.cpu().numpy()
            correct_values = correct_values.cpu().numpy()

            # Calculate metrics
            mse_loss = loss_fn(torch.tensor(predictions), torch.tensor(correct_values))
            rmse = torch.sqrt(mse_loss).item()
            mae = torch.mean(torch.abs(torch.tensor(correct_values) - torch.tensor(predictions))).item()
            mape = (torch.mean(torch.abs((torch.tensor(correct_values) - torch.tensor(predictions)) / torch.tensor(correct_values))) * 100).item()

            # Prepare the plot
            plt.figure(figsize=(12, 6))
            plt.plot(correct_values, label="Correct Values (Ground Truth)", color="blue")
            plt.plot(predictions, label="Predictions", color="red")
            plt.title("Model Predictions vs Correct Values")
            plt.xlabel("Time Step")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            plt.show()

        # Print metrics
        print(f"Metrics -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")

        # Return metrics as a dictionary
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }

        return metrics
    
    def torch_config(self):
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

        total_cpus = os.cpu_count()

        # Calculate 80% of the available CPUs
        num_threads = int(total_cpus * 0.8)

        print(f'Number of threads: {num_threads}.')

        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(num_threads)

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False       