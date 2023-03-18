import pickle
from flask import Flask, request, jsonify,app,render_template,url_for
import pandas as pd
import numpy as np

app=Flask(__name__,template_folder='template')

# Load the model
scaler=pickle.load(open(r'Deployment\boston_housing_deployment\scaler.pkl','rb'))
model=pickle.load(open(r'Deployment\boston_housing_deployment\model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])

def predict():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

if __name__=='__main__':
    app.run(debug=True)


