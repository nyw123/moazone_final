import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import random
import mlflow
import mlflow.sklearn
import pickle
import os
import subprocess
from pycaret.classification import *
from sklearn.metrics import log_loss
from sqlalchemy import create_engine


def load_from_s3(key):
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

def save_to_rds(df):
    engine = create_engine(f'mysql+pymysql://{DB_USER}:{DB_PW}@{DB_ADDR}:{DB_PORT}/{DB_NAME}')
    conn = engine.connect()
    df.to_sql(name='new', con=engine, if_exists='append',index=False)

    print('\n####### saved result to rds #######')

def save_to_s3():
    mlruns_direc = "./mlruns/"
    output = subprocess.run(["aws", "s3", "sync", "{}".format(mlruns_direc), "s3://{}".format(AWS_S3_BUCKET)], stdout=subprocess.PIPE, encoding='utf-8')
    print('\n####### saved model,metric to s3 #######')

def main():
    train_key = "upload/gcp.csv"
    train = load_from_s3(train_key)
    clf = setup(train, target = 'credit', train_size = 0.75, log_experiment = True, log_profile=True,log_plots = True,experiment_name='card-0902',fold=5)
    top = compare_models(sort='Accuracy', n_select=1)
    print('\n####### compare model  #######')
    tuned = tune_model(top)
    print('\n####### tuned Hyperparameter #######')
    final_model = finalize_model(tuned)
    print('\n####### finalized model  #######')
    prediction = predict_model(final_model, train)
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
    
    main()

