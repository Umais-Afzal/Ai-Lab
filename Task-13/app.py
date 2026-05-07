# app.py
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model and encoders
model = joblib.load('house_price_model.pkl')
encoders = joblib.load('label_encoders.pkl')

# Load dataset again only to get the exact feature names (excluding 'price' and 'date')
df_features = pd.read_csv('data.csv')
feature_names = df_features.drop(columns=['price', 'date']).columns.tolist()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_dict = {}
        for col in feature_names:
            val = data.get(col)
            if val is None:
                return jsonify({'error': f'Missing feature: {col}'}), 400

            # If the column was categorical, encode the input value
            if col in encoders:
                le = encoders[col]
                str_val = str(val)
                if str_val in le.classes_:
                    input_dict[col] = le.transform([str_val])[0]
                else:
                    # Fallback: use the first known category
                    input_dict[col] = le.transform([le.classes_[0]])[0]
            else:
                # Numeric feature
                input_dict[col] = float(val)

        input_df = pd.DataFrame([input_dict])
        prediction = model.predict(input_df)[0]
        return jsonify({'predicted_price': round(prediction, 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)