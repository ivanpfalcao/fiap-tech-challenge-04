BASEDIR="$( cd "$( dirname "${0}" )" && pwd )"

mlflow ui --backend-store-uri ${BASEDIR}/mlflow_data