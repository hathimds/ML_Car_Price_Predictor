import pickle
import os
from flask import Flask, request, jsonify, app, url_for, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from catboost import CatBoostRegressor


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__) # create an app instance

# Build full paths to your model files
model_path = os.path.join(BASE_DIR, 'models', 'Model.pkl')
brand_encoder_path = os.path.join(BASE_DIR, 'models', 'Brand_Encoder.pkl')
model_encoder_path = os.path.join(BASE_DIR, 'models', 'Model_Encoder.pkl')
onehot_encoder_path = os.path.join(BASE_DIR, 'models', 'OneHot_Encoder.pkl')

# Load models/encoders
model = pickle.load(open(model_path, 'rb'))
Brand_Encoder = pickle.load(open(brand_encoder_path, 'rb'))
Model_Encoder = pickle.load(open(model_encoder_path, 'rb'))
OneHot_Encoder = pickle.load(open(onehot_encoder_path, 'rb'))


@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract form data
        Brand = request.form['Brand'] 
        Model = request.form['Model'] 
        Fuel = request.form['Fuel'] 
        Transmission = request.form['Transmission'] 
        Year = int(request.form['Year']) 
        EngineSize = float(request.form['EngineSize']) 
        Mileage = int(request.form['Mileage']) 
        Doors = int(request.form['Doors']) 
        OwnerCount = int(request.form['OwnerCount']) 

        # Create a DataFrame for the input data
        input_df = pd.DataFrame({
            'Brand': [Brand],
            'Model': [Model],
            'Year': [Year],
            'EngineSize': [EngineSize],
            'Fuel': [Fuel],
            'Transmission': [Transmission],
            'Mileage': [Mileage],
            'Doors': [Doors],
            'OwnerCount': [OwnerCount]
        })
        
        # Encode Brand and Model using mean target encoding
        input_df['Encoded_Brand'] = input_df['Brand'].map(Brand_Encoder)
        input_df['Encoded_Model'] = input_df['Model'].map(Model_Encoder)
        input_df['Encoded_Brand'].fillna(input_df['Encoded_Brand'].mean(), inplace=True)
        input_df['Encoded_Model'].fillna(input_df['Encoded_Model'].mean(), inplace=True)
        input_df.drop(['Brand', 'Model'], axis=1, inplace=True)
        
        # One-hot encode Fuel and Transmission
        categorical_cols = ['Fuel', 'Transmission']
        encoded_array = OneHot_Encoder.transform(input_df[categorical_cols])
        encoded_df = pd.DataFrame(encoded_array, columns=OneHot_Encoder.get_feature_names_out(categorical_cols))
        
        # Merge encoded columns with input data
        input_df_encoded = input_df.drop(columns=categorical_cols).reset_index(drop=True)
        input_data = pd.concat([input_df_encoded, encoded_df], axis=1)
        
        # Make prediction
        prediction = model.predict(input_data)
        output = round(prediction[0], 2)
        
        if output < 0:
            return render_template('index.html', prediction_texts="Sorry you cannot sell this car")
        else:
            return render_template('index.html', prediction_text="Car is worth at: $ {}".format(output))
    else:
        return render_template('index.html')
        
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  
    app.run(host="0.0.0.0", port=port)
