BASEDIR="$( cd "$( dirname "${0}" )" && pwd )"

NAMESPACE="datalake-ns"

kubectl -n ${NAMESPACE} delete deployment stock-predictions

kubectl -n ${NAMESPACE} apply -f "${BASEDIR}/stock_predictions.yaml"