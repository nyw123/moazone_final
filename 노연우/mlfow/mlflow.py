import warnings
warnings.filterwarnings('ignore')
import subprocess
import pandas as pd
import numpy as np
import random
import mlflow
import mlflow.sklearn
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

def load_data(key):
    data = pd.read_csv(
        f"s3://{AWS_S3_BUCKET}/{key}",
        storage_options={
            "key": AWS_ACCESS_KEY_ID,
            "secret": AWS_SECRET_ACCESS_KEY
        },
    )
    data = data.drop(['index'], axis=1)
    data.fillna('NAN', inplace=True) 
    print('####### load data #######')
    return data

def set_xy(train):
    X = train.iloc[:,:-1]
    y = train.iloc[:,-1]
    print('####### set X,y #######')
    return X,y

def preprocess(X):
    X = pd.get_dummies(X)
    X = X.astype(float)
    print('####### preprocessing #######')
    return X

def tt_split(X,y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    print('####### train_test_split #######')
    return x_train, x_test, y_train, y_test

def make_model_exep(expname):
    forest = RandomForestClassifier(n_estimators=100)
    mlflow.set_experiment(expname)
    return forest

def train_save_model(x_train,y_train,x_test,y_test,model):
    uid = ''
    with mlflow.start_run():
        print('####### start mlflow #######')
        mlflow.autolog()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        metrics.accuracy_score(y_test, y_pred)
        uid = mlflow.active_run().info.run_uuid
    mlflow.end_run()
    print(f"####### uid : {uid} #######'")
    return uid

def load_model(uid):
    logged_model = f'runs:/{uid}/model'
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    print('####### load_model #######')
    return loaded_model

def predict(X,model):
    pred = model.predict(X)
    X['label'] = pred
    X.rename(columns={'Unnamed: 0':'index'},inplace=True)
    print('####### predicted #######')
    return X

def save_result(df):
    engine = create_engine(f'mysql+pymysql://{DB_USER}:{DB_PW}@{DB_ADDR}:{DB_PORT}/{DB_NAME}')
    conn = engine.connect()
    df.to_sql(name='new', con=engine, if_exists='append',index=False)
    print('\n####### saved result to rds #######')

def save_to_s3():
    mlruns_direc = "./mlruns/"
    output = subprocess.run(["aws", "s3", "sync", "{}".format(mlruns_direc), "s3://{}".format(AWS_S3_BUCKET)], stdout=subprocess.PIPE, encoding='utf-8')
    print(output.stdout)
    print('\n####### saved model to s3 #######')

def main():
    train = load_data(train_key)
    #test = load_data(test_key)
    X,y = set_xy(train)
    X = preprocess(X)

    x_train, x_test, y_train, y_test = tt_split(X,y)
    expname = 'card'
    model = make_model_exep(expname)
    uid = train_save_model(x_train,y_train,x_test,y_test,model)

    loaded = load_model(uid)
    df = predict(X,model)
    save_result(df)
    save_to_s3()
    
if __name__ == "__main__":
    AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

    DB_USER = os.getenv("DB_USER")
    DB_PW = os.getenv("DB_PW")
    DB_ADDR = os.getenv("DB_ADDR")
    DB_PORT = os.getenv("DB_PORT")
    DB_NAME = os.getenv("DB_NAME")
    
    train_key = "input/train.csv"
    #train_key2 = "upload/gcp.csv"
    test_key = "input/test.csv"

    main()
