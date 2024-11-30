#!/bin/bash

BASEDIR="$( cd "$( dirname "${0}" )" && pwd )"
NAMESPACE="datalake-ns"

while true; do
	echo "Starting port-forwarding..."
	kubectl -n "${NAMESPACE}" port-forward svc/mlflow-server-tracking 5000:80
	
	if [ $? -ne 0 ]; then
		echo "Port-forwarding failed. Retrying in 5 seconds..."
		sleep 5
	else
		echo "Port-forwarding stopped. Exiting loop."
		break
	fi
done