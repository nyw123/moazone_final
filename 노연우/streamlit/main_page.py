# import streamlit as st
# import pandas as pd
# import pandas_profiling
# import streamlit.components.v1 as components
# import plotly.figure_factory as ff
# import numpy as np
# import mlflow
# import sys

# st.markdown("# Main page 🎈")
# st.sidebar.markdown("# Main page 🎈")

# col1, col2, col3 = st.columns(3)


import time  
import pymysql
import numpy as np  
import pandas as pd 
import plotly.express as px  
import streamlit as st 
import psycopg2
import s3fs
import os
import mysql.connector


st.set_page_config(
    page_title="Real-Time Data Science Dashboard",
    page_icon="✅",
    layout="wide",
)
fs = s3fs.S3FileSystem(anon=False)

@st.experimental_memo(ttl=600)
def read_file(filename):
    df = pd.read_csv(fs.open(f'{filename}', mode='rb', index_col = 0))
    return df

df = read_file('card-s3/upload/gcp.csv')
df.fillna('Nan',inplace=True)
df = df.iloc[:,1:]
# dashboard title
st.title("Real-Time / Live Data Science Dashboard")

# top-level filters
job_filter = st.selectbox("Select the Job", pd.unique(df["occyp_type"]))

# creating a single-element container
placeholder = st.empty()

# dataframe filter
df = df[df["occyp_type"] == job_filter]

@st.experimental_singleton
def init_connection():
    return mysql.connector.connect(**st.secrets["mysql"])

conn = init_connection()


@st.experimental_memo(ttl=600)
def run_query(query):
    with conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()


with placeholder.container():


    rows = run_query("SELECT accuracy,f1 from metrics top 2;")

    before = rows[0]
    now  = rows[1]

    before_acc, before_f1 = before[0] , before[1]
    now_acc, now_f1 = now[0] , now[1]

    acc_delta = now_acc - before_acc
    f1_delta = now_f1 - before_f1

    rows2 = run_query("SELECT count(*) from data;")
    data_count = rows2[0]
    rows3 = run_query("SELECT count(*) from new_data;")
    new_data_count = rows3[0]


    # create three columns
    kpi1, kpi2, kpi3 = st.columns(3)

    # fill in those three columns with respective metrics or KPIs
    kpi1.metric(
        label="Accuracy",
        value=now_acc,
        delta= acc_delta,
    )
    
    kpi2.metric(
        label="F1-score",
        value=now_f1,
        delta=f1_delta,
    )
    
    kpi3.metric(
        label="New Data",
        value=data_count,
        delta=new_data_count,
    )

    fig_col1, fig_col2 = st.columns(2)

    with fig_col1:
        st.markdown("### First Chart")
        fig = px.density_heatmap(
            data_frame=df, y="income_total", x="income_type", 
            marginal_x='histogram', marginal_y='histogram'
        )
        st.write(fig)
        
    with fig_col2:
        st.markdown("### Second Chart")
        fig2 = px.bar(df, x="edu_type", y="income_total")

        st.write(fig2)

    st.markdown("### Detailed Data View")

    st.dataframe(df)
