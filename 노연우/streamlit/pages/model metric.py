import streamlit as st
import mlflow
from pycaret.classification import *
import pandas as pd

st.markdown("# model metric 🎉")
st.sidebar.markdown("# model metric 🎉")


# @st.cache(allow_output_mutation=True)
# def create_model_cache(estimator):
#     return create_model(estimator)


train = pd.read_csv('../train.csv')
train = train.iloc[:2000,:]
clf = setup(train, target = 'label', train_size = 0.7,experiment_name='card-0910',silent = True)
top = compare_models(sort='Accuracy', n_select=1)
print('\n####### compare model  #######')

tuned = tune_model(top)
print('\n####### tuned Hyperparameter #######')
final_model = finalize_model(top)
print('\n####### finalized model  #######')

#final_model = load_model('./mlruns/1/6da70823adeb41ffab174101295d4838/artifacts/model/model') 
plot_model(final_model, plot = 'confusion_matrix', display_format="streamlit")
plot_model(final_model, plot = 'auc', display_format="streamlit")
#plot_model(final_model, plot = 'threshold', display_format="streamlit")
plot_model(final_model, plot = 'pr', display_format="streamlit")
plot_model(final_model, plot = 'error', display_format="streamlit")
plot_model(final_model, plot = 'class_report', display_format="streamlit")
plot_model(final_model, plot = 'boundary', display_format="streamlit")
#plot_model(final_model, plot = 'rfe', display_format="streamlit")
plot_model(final_model, plot = 'learning', display_format="streamlit")
#plot_model(final_model, plot = 'manifold', display_format="streamlit")
#plot_model(final_model, plot = 'calibration', display_format="streamlit")
plot_model(final_model, plot = 'vc', display_format="streamlit")
plot_model(final_model, plot = 'dimension', display_format="streamlit")
plot_model(final_model, plot = 'feature', display_format="streamlit")
plot_model(final_model, plot = 'feature_all', display_format="streamlit")
plot_model(final_model, plot = 'parameter', display_format="streamlit")
#plot_model(final_model, plot = 'lift', display_format="streamlit")
#plot_model(final_model, plot = 'gain', display_format="streamlit")
#plot_model(final_model, plot = 'tree', display_format="streamlit")
#plot_model(final_model, plot = 'ks', display_format="streamlit")