from flask import Flask, render_template, request
from impala.dbapi import connect
import mlflow
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

app = Flask(__name__)


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
    #raw_data='{"Age": 27, "Vintage": 26, "Avg_Account_Balance": 707906, "Channel_Code": "X1", "Credit_Product": "No", "Gender": "Female", "Is_Active": "No", "Occupation": "Salaried", "Region_Code": "RG256"}'  
    
    #data = json.loads(raw_data)
    #print(f"json raw_data = {[raw_data]}")

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

PORT = os.getenv('CDSW_APP_PORT', '8090')
print(f"Port: {PORT}")
print(f"Listening on Port: {PORT}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
      id = request.form['id']
        #id='NNVBBKZB'

        # Query the database
        # ## jdbc:impala://dwarehouse-gateway.jqrhv9ch.l39n-29js.a8.cloudera.site:443/;ssl=1;transportMode=http;httpPath=dwarehouse/cdp-proxy-api/impala;AuthMech=3;

        #conn = connect(host='your_impala_host', port=21050)

      def conn():
        return connect(host='dwarehouse-gateway.jqrhv9ch.l39n-29js.a8.cloudera.site', port=443, \
                              timeout=None, use_ssl=True, ca_cert=None, database='jvp_cc_lead_model',\
                              user="csso_jprosser", password="*******", kerberos_service_name='impala', \
                              auth_mechanism="LDAP", krb_host=None, use_http_transport=True, \
                              http_path='cliservice', http_cookie_names=None, retries=3, jwt=None,\
                              user_agent=None)

      engine = db.create_engine('impala://', creator=conn)


      with engine.connect() as conn:
        query = f"""
          SELECT Age, Vintage, Avg_Account_Balance, Channel_Code, Credit_Product, Gender, Is_Active ,Occupation, Region_Code 
          FROM jvp_cc_lead_model.cc_lead_train  
          WHERE ID = '{id}'
          """
        print(f"query: {query}")

        result = conn.execute(text(query)).fetchone()
        print(f"back from query: {result}")
 
        conn.close()

        if result:
            # Store results in a dictionary
            data = {
               
                "Age": result[0],
                "Vintage": result[1],
                "Avg_Account_Balance": result[2],
                "Channel_Code": result[3],
                "Credit_Product": result[4],
                "Gender": result[5],
                "Is_Active": result[6],
                "Occupation": result[7],
                "Region_Code": result[8]
            }
            print(f"data dict: {data}")

            # Normalize the data
            normalized_data = normalize_data(data)
            print(f"Normalized_data: {[normalized_data]}")
            #cc_vector_df = pd.DataFrame([normalized_data])
            #print(f"Vector df = {normalized_data}")
            #cc_vector='{"Age":0.6831522461,"Vintage":-0.8637453118,"Avg_Account_Balance":-0.1245877441,"Channel_Code":1.0,"Credit_Product":2.0,"Gender":1.0,"Is_Active":0.0,\
            #"Occupation":3.0,"Region_Code":33.0}'          
            #data = json.loads(cc_vector)
            #cc_vector_df = pd.DataFrame([data])        
            #print(f"Vector df = {cc_vector_df}")

            # Make prediction
            prediction = rf_loaded.predict(normalized_data)
            print(f"prediction = {prediction[0]}")
            return render_template('result.html', prediction=prediction[0])
        else:
            return render_template('index.html', error="No data found for this ID")

    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=PORT)
    



