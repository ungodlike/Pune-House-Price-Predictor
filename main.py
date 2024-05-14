import pandas as pd
import pickle
import numpy as np
from flask import Flask, render_template, request
from flask_cors import CORS
import os

#os.chdir(os.path.dirname(__file__))



app = Flask(__name__)
CORS(app) #manage cors

data = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open("RidgeRegModel.pkl", 'rb')) #read binary method
 
 
@app.route('/')
def index():
    
    site_locations = sorted(data['site_location'].unique())
    return render_template('index.html', site_locations=site_locations)

@app.route('/predict', methods=['POST'])
def predict():
    site_location = request.form.get('site_location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')
    
    print(site_location, bhk, bath, sqft)
    
    input = pd.DataFrame([[site_location,sqft,bath,bhk]] ,columns=['site_location', 'total_sqft', 'bath', 'bhk'])
    prediction = pipe.predict(input)[0] * 1e5 
   
    return str(np.round(prediction, 2)) 

if __name__ == "__main__":
    app.run(debug=True, port=os.getenv("PORT", default=5000))
