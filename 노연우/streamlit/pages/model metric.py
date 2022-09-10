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
train = train.iloc[:1000,:]
clf = setup(train, target = 'label', train_size = 0.75, log_experiment = True,experiment_name='card-0910',fold=5,silent = True,ignore_features=['index'])
top = compare_models(sort='Accuracy', n_select=1)
print('\n####### compare model  #######')

#tuned = tune_model(top)
#print('\n####### tuned Hyperparameter #######')
final_model = finalize_model(top)
print('\n####### finalized model  #######')

plot_model(final_model, plot = 'auc', use_train_data = True, display_format="streamlit")
plot_model(final_model, plot = 'confusion_matrix', display_format="streamlit")
plot_model(final_model, plot = 'feature', display_format="streamlit")
