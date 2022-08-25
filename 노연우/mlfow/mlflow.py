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

AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

DB_USER = os.getenv("DB_USER")
DB_PW = os.getenv("DB_PW")
DB_ADDR = os.getenv("DB_ADDR ")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

train_key = "input/train.csv"
train_key2 = "upload/gcp.csv"
test_key = "input/test.csv"


train = pd.read_csv(
    f"s3://{AWS_S3_BUCKET}/{train_key2}",
    storage_options={
        "key": AWS_ACCESS_KEY_ID,
        "secret": AWS_SECRET_ACCESS_KEY
    },
)

print('####### load train #######')
'''
test = pd.read_csv(
    f"s3://{AWS_S3_BUCKET}/{test_key}",
    storage_options={
        "key": AWS_ACCESS_KEY_ID,
        "secret": AWS_SECRET_ACCESS_KEY
    },
)

print('####### load test ####### ')
'''
train = train.drop(['index'], axis=1)
#test = test.drop(['index'], axis=1)

train.fillna('NAN', inplace=True) 
#test.fillna('NAN', inplace=True)

x = train.iloc[:,:-1]
y = train.iloc[:,-1]

print('####### set X,y #######')

X = pd.get_dummies(x)

X = X.astype(float)

print('####### preprocessing #######')
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print('####### split #######')

forest = RandomForestClassifier(n_estimators=100)
mlflow.set_experiment('card')
uid = ''
with mlflow.start_run():
    print('####### start mlflow #######')
    mlflow.autolog()
    forest.fit(x_train, y_train)
    y_pred = forest.predict(x_test)
    metrics.accuracy_score(y_test, y_pred)
    uid = mlflow.active_run().info.run_uuid

mlflow.end_run()

print('####### load model and test moa  #######')

moa = X.iloc[100,:].to_frame().T
ans = int(train.iloc[100,-1])
logged_model = f'runs:/{uid}/model'

loaded_model = mlflow.pyfunc.load_model(logged_model)

print('predict : '+ str(int(loaded_model.predict(moa))))
print('answer : ' + str(ans))


pred = forest.predict(X)
x['label'] = pred
x.rename(columns={'Unnamed: 0':'index'},inplace=True)
engine = create_engine(f'mysql+pymysql://{DB_USER}:{DB_PW}@{DB_ADDR}:{DB_PORT}/{DB_NAME}')
conn = engine.connect()
x.to_sql(name='new', con=engine, if_exists='append',index=False)

print('\n####### save label to rds #######')

s3_bucket_name = 'card-s3'
mlruns_direc = "./mlruns/"
output = subprocess.run(["aws", "s3", "sync", "{}".format(mlruns_direc), "s3://{}".format(s3_bucket_name)], stdout=subprocess.PIPE, encoding='utf-8')
print(output.stdout)
print('\n####### save model to s3 #######')
