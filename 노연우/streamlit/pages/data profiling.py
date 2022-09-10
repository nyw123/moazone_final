import streamlit as st
import pandas as pd
import pandas_profiling
import streamlit.components.v1 as components
import plotly.figure_factory as ff
import numpy as np
import glob

path = "./pages/*"
file_list = glob.glob(path)
file_list_html = [file for file in file_list if file.endswith(".html")]

st.markdown("# data profiling ❄️")
st.sidebar.markdown("# data profiling ❄️")
#print(file_list)
if len(file_list_html)>0:
    HtmlFile = open("./pages/profile.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    components.html(source_code,width=1000,height = 800,scrolling=True)

else:
    df = pd.read_csv('../train.csv')
    profile = df.profile_report()
    profile.to_file(output_file="./pages/profile.html")
    HtmlFile = open("../profile.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    components.html(source_code,width=1000,height = 800,scrolling=True)