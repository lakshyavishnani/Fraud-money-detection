from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    prediction = model.predict([data])
    return jsonify({'fraud': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)