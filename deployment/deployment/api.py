@app.route('/predict', methods=['POST'])
def predict_air_quality():
    data = request.json
    prediction = model.predict(data)
    confidence = model.predict_proba(data)
    return jsonify({
        'pm25_prediction': prediction,
        'confidence_interval': confidence,
        'timestamp': datetime.now()
    })