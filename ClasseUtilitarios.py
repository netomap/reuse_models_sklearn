import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer, MinMaxScaler
import joblib

class Utilitarios:

    """
    Essa classe tem funções que são úteis para a transformação bruta dos dados para o formato necessário
    que o modelo fará as inferências. 
    """

    def __init__(self):
        print ('iniciado')
    
    def import_parameters(self, joblib_model_path):
        """
        Importando o documento que tem os parâmetros necessários para auxiliar o modelo.
        """
        obj_joblib = joblib.load(joblib_model_path)
        self.model = obj_joblib['model']
        self.label_encoders = obj_joblib['label_encoders']
        self.label_binarizers = obj_joblib['label_binarizers']
        self.scaler = obj_joblib['scaler']
        self.encoder_output = obj_joblib['encoder_output']
        print ('modelo importado')
        print (obj_joblib['description'])


    def pipeline(self, df_, label_encoders_, label_binarizers_):
        """
        Primeiro pegamos todas colunas que são numéricas, que não estão no label_encoders e label_binarizers e jogamos para novo_df
        Depois pegamos todas do label_encoders e transformamos os dados e anexamos no novo_df
        Por último, fazemos a transformação do label_binarizers colocando os nomes das colunas como nome_atributo
        e anexamos por útltimo no novo_df.
        """
        novo_df = pd.DataFrame() # dataframe zerado

        for nome_coluna in df_.columns.values: # aqui pegamos todas não categóricas
            if (nome_coluna not in label_encoders_ and nome_coluna not in label_binarizers_):
                novo_df[nome_coluna] = df_[nome_coluna]

        for nome_coluna, encoder_coluna in label_encoders_.items():  #aqui fazemos as simples transformação com 
            novo_df[nome_coluna] = encoder_coluna.transform(df_[nome_coluna])  # LabelEncoders
        
        for nome_coluna, encoder_coluna in label_binarizers_.items():
            nomes_colunas_oht = [nome_coluna + '_' + l for l in encoder_coluna.classes_]
            df_temp = pd.DataFrame(encoder_coluna.transform(df_[nome_coluna]), columns=nomes_colunas_oht)
            novo_df = pd.concat([novo_df, df_temp], axis=1, join='inner')
        
        return novo_df
    
    def predict_function(self, vetor_teste):
        """
        Função final, que pega um vetor e faz todo o processo de transformação até predição e resposta final.
        """
        colunas = ['age', 'job', 'marital', 'education', 'default', 'balance',
        'housing', 'loan', 'contact', 'day', 'month', 'duration',
        'campaign', 'pdays', 'previous', 'poutcome']
        novo_df = pd.DataFrame(vetor_teste, columns=colunas)  # transformando o vetor em df
        df_transformado = self.pipeline(novo_df, self.label_encoders, self.label_binarizers) # pipeline do df inicial
        df_transformado = self.scaler.transform(df_transformado)   # normalizando os dados
        y_pred = self.model.predict(df_transformado)   # predição
        y_prob = self.model.predict_proba(df_transformado) # predição com probabilidades

        y_pred_label = self.encoder_output.inverse_transform(y_pred)   # pega o valor de acordo com o encoder_output
        response = []
        for label, vetor_probs, indice_pred in zip(y_pred_label, y_prob, y_pred):
            response.append({'label': label, 'prob': vetor_probs[indice_pred]})

        return response