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
# col1.metric("Accuracy", "0.6878", "0.05")
# col2.metric("F1-score", "0.5974", "-0.002")
# col3.metric("Total Data", "31204", "New 403")


import time  # to simulate a real time data, time loop
import pymysql
import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development
import psycopg2
import s3fs
import os



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

# near real-time / live feed simulation
for seconds in range(1):

    with placeholder.container():

        # create three columns
        kpi1, kpi2, kpi3 = st.columns(3)

        # fill in those three columns with respective metrics or KPIs
        kpi1.metric(
            label="Accuracy",
            value=0.6921,
            delta= 0.005,
        )
        
        kpi2.metric(
            label="F1-score",
            value=0.6553,
            delta=-0.012,
        )
        
        kpi3.metric(
            label="New Data",
            value=39424,
            delta=130,
        )

        # create two columns for charts
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
            #animation_frame="year", animation_group="country", range_y=[0,4000000000])
            #fig2.show()

            st.write(fig2)

        st.markdown("### Detailed Data View")
        st.dataframe(df)


#st.dataframe(df)