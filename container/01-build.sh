BASEDIR="$( cd "$( dirname "${0}" )" && pwd )"

docker build \
    -f "${BASEDIR}/stock_predictions.dockerfile" \
    -t "ivanpfalcao/stock_predictions:1.0.0" \
    --progress=plain \
    "${BASEDIR}/.."