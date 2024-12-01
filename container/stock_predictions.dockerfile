FROM python:3.10.15-slim-bookworm

SHELL ["/bin/bash", "-c"]
WORKDIR /app



ENV APP_DIR="/app/stock_predictions"

ENV PYTHON_VENV="/app/venv"

COPY ./stock_predictions/requirements.txt ${APP_DIR}/

RUN python -m venv ${PYTHON_VENV} \
    && source "${PYTHON_VENV}/bin/activate" \
    && pip install -r ${APP_DIR}/requirements.txt

COPY ./stock_predictions ./stock_predictions

RUN source "${PYTHON_VENV}/bin/activate" \
    && bash "${APP_DIR}/install.sh"

EXPOSE 8000

ENTRYPOINT source "${PYTHON_VENV}/bin/activate" \
            && bash "${APP_DIR}/run.sh"





