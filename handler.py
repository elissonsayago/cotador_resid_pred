import pandas as pd
import pickle
from flask import Flask, request
import os

#load model
model=pickle.load(open('svm_model_cotador.pkl','rb'))

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
    

    #convertendo as colunas para o tipo original correto
    df_raw['COD_CORRETOR'] = df_raw['COD_CORRETOR'].astype(float)
    df_raw['FLG_OPERACAO_ESPECIAL'] = df_raw['FLG_OPERACAO_ESPECIAL'].astype(object)
    df_raw['COD_TIPO_SEGURO'] = df_raw['COD_TIPO_SEGURO'].astype(float)
    df_raw['COD_GRUPO_ATIVIDADE'] = df_raw['COD_GRUPO_ATIVIDADE'].astype(float)
    df_raw['COD_TIPO_CONSTRUCAO'] = df_raw['COD_TIPO_CONSTRUCAO'].astype(float)
    df_raw['FLG_POSSUI_CRITICA'] = df_raw['FLG_POSSUI_CRITICA'].astype(object)
    df_raw['FLG_EXIGE_VISTORIA'] = df_raw['FLG_EXIGE_VISTORIA'].astype(object)
    df_raw['VAL_PREMIO_AVISTA'] = df_raw['VAL_PREMIO_AVISTA'].astype(float)
    df_raw['COD_FORMA_PAGTO'] = df_raw['COD_FORMA_PAGTO'].astype(float)


    #prediction
    pred = model.predict ( df_raw )
    df_raw['PREDICTION_FLG_PROPOSTA'] = pred
    return df_raw.to_json(orient='records')

if __name__ == '__main__':
    #start flask
    #app.run( host='0.0.0.0', port='5000')
    port=os.environ.get('PORT',5000)
    app.run( host='0.0.0.0', port=port)