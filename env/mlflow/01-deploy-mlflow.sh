BASEDIR="$( cd "$( dirname "${0}" )" && pwd )"

NAMESPACE="datalake-ns"

kubectl create namespace "${NAMESPACE}"
helm -n ${NAMESPACE} install mlflow-server -f "${BASEDIR}/values.yaml" oci://registry-1.docker.io/bitnamicharts/mlflow