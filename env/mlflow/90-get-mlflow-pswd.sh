BASEDIR="$( cd "$( dirname "${0}" )" && pwd )"

NAMESPACE="datalake-ns"
  
echo Username: $(kubectl get secret --namespace ${NAMESPACE} mlflow-server-tracking -o jsonpath="{ .data.admin-user }" | base64 -d)
echo Password: $(kubectl get secret --namespace ${NAMESPACE} mlflow-server-tracking -o jsonpath="{.data.admin-password }" | base64 -d)