BASEDIR="$( cd "$( dirname "${0}" )" && pwd )"

MLFLOW_TRACKING_URL="http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME="Stock Prediction"
MLFLOW_MODEL_RUN_ID="runs:/8e8309d5af914d9b99263dfcccd0e4e2/model"

docker rm -f stock_predictions

docker run -d \
	-p 8000:8000 \
	-e "MLFLOW_TRACKING_URL=${MLFLOW_TRACKING_URL}" \
	-e "MLFLOW_EXPERIMENT_NAME=${MLFLOW_EXPERIMENT_NAME}" \
    -e "MLFLOW_MODEL_RUN_ID=${MLFLOW_MODEL_RUN_ID}" \
    --network=host \
	--name stock_predictions \
	"ivanpfalcao/stock_predictions:1.0.0"  