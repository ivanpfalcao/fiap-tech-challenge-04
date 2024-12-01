from stock_predictions.predictors.stock_predictions import PredictionRunner


from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
from pydantic import BaseModel
import uvicorn
from stock_predictions.predictors.stock_predictions import PredictionRunner


app = FastAPI(
	title="Stock Prediction App",
	description="A FastAPI app with best practices for configuration",
	version="1.0.0",
	docs_url="/docs",
	redoc_url="/redoc",
	openapi_url="/openapi.json"
)



pred_runner = PredictionRunner(tracking_url="http://localhost:5000", experiment_name="Stock Prediction")

pred_runner.torch_config()

pred_runner.load_model_mlflow("runs:/388a859230994515bf04759f769c7668/model")

# Error handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
	return JSONResponse(
		status_code=HTTP_422_UNPROCESSABLE_ENTITY,
		content={"detail": exc.errors(), "body": exc.body},
	)

# Custom error handler for 404
@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
	return JSONResponse(
		status_code=exc.status_code,
		content={"message": exc.detail},
	)

# Data validation using Pydantic
class DataList(BaseModel):
	data_list: list[float]

@app.post("/predict/")
async def predict(data_list: DataList):
    print(data_list.data_list)
    prediction = pred_runner.predict(data_list.data_list)
    return {"prediction": prediction, "message": "Prediction Ok"}

# Root endpoint
@app.get("/")
async def read_root():
	return {"message": "Welcome to FastAPI!"}

# Health check endpoint
@app.get("/health/")
async def health_check():
	return {"status": "ok"}

if __name__ == "__main__":
	uvicorn.run(
		"main:app",
		host="0.0.0.0",
		port=8000,
		reload=True,
		workers=2,
	)
