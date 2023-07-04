import os
import pandas as pd
import pickle
from flask import Flask, request
from cotador_resid_pred import CotadorResidPred

#load model
model=pickle.load(open('model_cotador.pkl','rb'))

#instantiate Flask
app = Flask ( __name__)

@app.route('/predict', methods=['POST'])
def predict():
    test_json=request.get_json()
    
    #colect data
    if test_json:
        if isinstance (test_json, dict):
            df_raw = pd.DataFrame( test_json, index[0])
        else:
            df_raw = pd.DataFrame (test_json, columns=test_json[0].keys())
    

    #instantiate data preparation
    pipeline = CotadorResidPred()

    #data preparation
    df1 = pipeline.datapreparation(df_raw)

    #prediction
    pred = model.predict ( df1 )
    df1['prediction'] = pred
    return df1.to_json(orient='records')

if __name__ == '__main__':
    #start flask
    port=os.environ.get('PORT',5000)
    app.run( host='0.0.0.0', port=port)