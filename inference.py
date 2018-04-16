#*-coding:utf-8 -*-
#######
import os
import os.path
import requests
import pandas as pd 
import numpy as np 
from sklearn.externals import joblib###
from sklearn.metrics import accuracy_score###
from flask import Flask, jsonify
from flask import request


app = Flask(__name__)

@app.route("/predict", methods=['GET','POST'])
def query_prediction():
    
    ## Step 1, read the source data
    
    if request.method == 'POST':
        data=request.get_json()
        data=data['data']
    
    X_ret = read_sample_db(data)
    
    ## Step2, load model and start to predict
    
    predict_data = load_sample_model('/inference_engine/model.pkl', X_ret)
    
    ## Step3, write the predicted data into destination
    
    write_model_info(predict_data)
    
    return jsonify({'message': 'predict success'})


def load_sample_model(filepath, training_data):
    
    clf = joblib.load(filepath)
    y_pre = clf.predict(training_data)
    return y_pre
    pass

def read_sample_db(df):
    
    data1 = pd.DataFrame(df)
    col = [ 'STATUS_FAN',

            'VOLTAGE_INPUT',

            'PRESSURE_OUTPUT',

            'KW_FAN',       

            'KW_EQUIPMENT',      

            'KW_SUMMARY'          
         ]
    data1 = data1[col]
    data1 = np.array(data1)
    X_test = data1.reshape([1, 60])
    return X_test
    pass

def write_model_info(pre):    
    if (os.path.exists("/inference_engine/predict_result.txt")==True):
        with open('/inference_engine/predict_result.txt','a') as df2:
            df2.write(",")
            df2.write(pre[0])
    elif (os.path.exists("/inference_engine/predict_result.txt")==False):
        with open('/inference_engine/predict_result.txt','w') as df1:
            df1.write(pre[0])
    pass


port = os.getenv('PORT', '7500')

if __name__ == "__main__":
    app.run(host = '0.0.0.0', port = int(port))
