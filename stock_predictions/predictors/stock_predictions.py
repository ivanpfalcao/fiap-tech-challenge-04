import mlflow
import mlflow.pytorch
import numpy as np
import torch

from stock_predictions.predictors.ai import StockPrediction, StockModel


class PredictionRunner():
    def __init__(self, tracking_url = "http://localhost:5000", experiment_name="Stock Prediction"):
        mlflow.set_tracking_uri(tracking_url)
        mlflow.set_experiment(experiment_name)
        self.stock_p  = StockPrediction()          

    def torch_config(self):
        self.stock_p.torch_config()
        
    def model_training(self, enterprise = 'GOOGL', start_date = '2024-05-20', end_date = '2024-11-20'):
                     
        

        self.stock_p.load_yfinance_data(enterprise, start_date, end_date) 

        self.model = StockModel()

        x_train, y_train, x_test, y_test = self.stock_p.get_torch_train_n_test()

        self.stock_p.train_model(self.model, x_train, y_train, x_test, y_test, n_epochs = 100000, weight_decay=5e-05, log_frequency=5000, prefix=enterprise)

        self.stock_p.test_model(self.model, x_train, x_test)

    def load_model_mlflow(self, model_run_id):
        self.stock_p = StockPrediction()
        self.stock_p.load_model_from_mlflow(model_run_id)


    def test_mlflow_model(self, enterprise = 'GOOGL', start_date = '2022-01-01', end_date = '2022-12-31', lookback = 4):

        self.stock_p.load_yfinance_data(enterprise, start_date, end_date)
        
        x_train, y_train, x_test, y_test = self.stock_p.get_torch_train_n_test(lookback=lookback)

        # Use predict_and_plot
        metrics = self.stock_p.predict_and_plot(x_test, y_test)

        # Print metrics
        print("Metrics:", metrics)                   
                


    def predict(self, input_data: list):
        """
        Predict the output for the given input data using the loaded model.

        Args:
            input_data (list): Input data as a sequence of values.

        Returns:
            torch.Tensor: Model predictions.
        """
        if self.stock_p.model is None:
            raise ValueError("No model loaded. Please load a model first using 'load_model_mlflow'.")

        # Convert the input data to a 2D array (sequence_length, input_size=1)
        numpy_array = np.array(input_data, dtype=np.float32).reshape(-1, 1)

        # Convert the input to a 3D tensor (batch_size=1, sequence_length, input_size=1)
        input_tensor = torch.tensor(numpy_array).unsqueeze(0).to(self.stock_p.device)

        self.stock_p.model.eval()
        with torch.no_grad():
            predictions = self.stock_p.model(input_tensor)
        
        # Return predictions as a NumPy array
        return predictions.squeeze().cpu().numpy().tolist()