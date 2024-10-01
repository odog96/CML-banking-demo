from flask import Flask, render_template, request
from impala.dbapi import connect
import pandas as pd
import json
import os




import sqlalchemy as db
from sqlalchemy import Table, MetaData, select, func, text
import pickle
import numpy
#from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import joblib

ohefit = joblib.load("data/ohefit.save")

# Load the MLflow model
#model_path = "path/to/your/mlflow/model.pkl"
#loaded_model = mlflow.pyfunc.load_model(model_path)

#logged_model = '/home/cdsw/.experiments/b0se-2xuo-w64u-04yn/atgg-nm4g-tqs7-1r4h/artifacts/lgb_model'

# Load model
# Load the model from the file in another script/notebook
with open('models_from_code/random_forest_model.pkl', 'rb') as model_file:
    rf_loaded = pickle.load(model_file)


with open('data/standardscaler.pkl', 'rb') as file:
    loaded_standardscaler = pickle.load(file)

# with open('data/labelencoder.pkl', 'rb') as file:
#     loaded_labelencoder = pickle.load(file)

with open('data/data_num_cols.pkl', 'rb') as file:
    data_num_cols = pickle.load(file)

with open('data/data_cat_cols.pkl', 'rb') as file:
    data_cat_cols = pickle.load(file)

with open('data/onehotenc.pkl','rb') as file:
    ohefit = pickle.load(file)

def normalize_data(raw_data):
    print(f"top raw_data = {raw_data}\n")
    
    cc_vector_df = pd.DataFrame([raw_data])    
    print(f"raw pandas dataframe {cc_vector_df}\n")    
        
    data_num_data = cc_vector_df.loc[:, data_num_cols]
    data_cat_data = cc_vector_df.loc[:, data_cat_cols]
    
    #print("Shape of num data:", data_num_data.shape)
    #print("Shape of cat data:", data_cat_data.shape)
    
    print(f"data_num_data: {data_num_data}\n")
    print(f"data_cat_data: {data_cat_data}\n")

    data_num_data['Avg_Account_Balance'] = np.log(data_num_data['Avg_Account_Balance'])
        
    #data_num_data_s = loaded_standardscaler.fit_transform(data_num_data)
    data_num_data_s = loaded_standardscaler.transform(data_num_data)
    data_num_data_df_s = pd.DataFrame(data_num_data_s, columns = data_num_cols)

    #print('numerical test data post standard scaler', data_num_data_s)
   
    data_cat_data_norm = ohefit.transform(data_cat_data)    

    #print('columns of categorical encoded df', data_cat_data_norm.columns)
    #print('type of categorical encoded df', type(data_cat_data_norm))

    #print (' catagorical data post encoding', data_cat_data_norm)
  
    data_new = pd.concat([data_num_data_df_s, data_cat_data_norm], axis = 1)
    
    print(f"transformed data={data_new}\n")
    return data_new

@cdsw.model_metrics
def lead_prediction(args):
    print('args',args)
    print('type args',args)

    normalized_data = normalize_data(args)

    prediction = rf_loaded.predict(normalized_data)
    
    return args



