import pickle

class CotadorResidPred (object):
    def __init__ (self):
        self.VAL_PREMIO_LIQUIDO_scaler = pickle.load(open('VAL_PREMIO_LIQUIDO_scaler.pkl','rb'))
        self.VAL_PREMIO_AVISTA_scaler = pickle.load(open('VAL_PREMIO_AVISTA_scaler.pkl','rb'))
    
    def datapreparation(self, df):
        #rescaling
        df['VAL_PREMIO_LIQUIDO'] = self.VAL_PREMIO_LIQUIDO_scaler.transform(df[['VAL_PREMIO_LIQUIDO']].values)
        df['VAL_PREMIO_AVISTA'] = self.VAL_PREMIO_AVISTA_scaler.transform(df[['VAL_PREMIO_AVISTA']].values)
        
        return df