from flask import Flask, render_template, request, jsonify
from build_features import build_features
from load_models import models

app = Flask(__name__)


def model_predict(lst):
	return {}

@app.route('/predict', methods = ['POST'])
def predict():
	print(models.booster)
	try:
		data = request.json
		if data:
			X = build_features(data, models)
			# return jsonify({'prediction':models.booster.predict_proba(X).tolist()})
			return jsonify(models.booster.predict_proba(X)[:,1].tolist())
	except Exception as e:
		return {'error':str(e)}

# Run
if __name__ == '__main__':
	app.debug = True
	app.run(
			host = "0.0.0.0",
			port = 80
		)
