from typing import List, Union
import datetime

from fastapi import FastAPI
import joblib
import numpy as np
from prometheus_client import make_asgi_app, Counter, Histogram


MODEL = joblib.load("model.joblib")

predictions_counter = Counter('predictions', 'Number of predictions')
model_info_counter = Counter('model_info', 'Number of executions of model information')
predictions_output_hist = Histogram('predictions_output', 'Predictions output')
predictions_scores_hist = Histogram('predictions_scores', 'Predictions scores')
predictions_latency_hist = Histogram('predictions_latency', 'Latency of predictions')

app = FastAPI()
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.post("/prediction")
async def read_data(feature_vector: Union[List[float], List[List[float]]], score: bool):
	start = datetime.datetime.now()
	predictions_counter.inc()

	response = {}
	feature_vector = np.array(feature_vector)
	if feature_vector.ndim == 1:
		feature_vector = feature_vector.reshape(1, -1)

	pred = MODEL.predict(feature_vector).tolist()
	response["is_inlier"] = pred
	predictions_output_hist.observe(int(pred[0]))

	if score:
		mod_score = MODEL.score_samples(feature_vector).tolist()
		response["anomaly_score"] = mod_score
		predictions_scores_hist.observe(int(mod_score[0]))

	latency = datetime.datetime.now() - start
	predictions_latency_hist.observe(latency.total_seconds())
	return response


@app.get("/model_information")
async def get_info():
	model_info_counter.inc()
	return MODEL.get_params()
