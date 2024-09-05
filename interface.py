import streamlit as st
import numpy as np
import pandas as pd
from joblib import load
import time
ensemble = load(open(r"C:\Users\dell\OneDrive\Desktop\Ml interface\en1.joblib", 'rb'))
st.title("Welcome to SGPA predictor")
st.subheader("We are using a Ensemble regressor model that uses 2 base models")
st.subheader("1.MLR ")
st.subheader("2.Random Forest")
name=st.text_input("Your Name")
ml1,ml2=st.columns(2)
ml1=ml1.number_input("Enter Ml Mid1 marks",min_value=0,max_value=30,step=1)
ml2=ml2.number_input("Enter Ml Mid2 marks",min_value=0,max_value=30,step=1)
atcd1,atcd2=st.columns(2)
atcd1=atcd1.number_input("Enter ATCD Mid1 marks",min_value=0,max_value=30,step=1)
atcd2=atcd2.number_input("Enter ATCD Mid2 marks",min_value=0,max_value=30,step=1)
cn1,cn2=st.columns(2)
cn1=cn1.number_input("Enter CN Mid1 marks",min_value=0,max_value=30,step=1)
cn2=cn2.number_input("Enter CN Mid2 marks",min_value=0,max_value=30,step=1)
ipp1,ipp2=st.columns(2)
ipp1=ipp1.number_input("Enter IPP Mid1 marks",min_value=0,max_value=30,step=1)
ipp2=ipp2.number_input("Enter IPP Mid2 marks",min_value=0,max_value=30,step=1)
ann1,ann2=st.columns(2)
ann1=ann1.number_input("Enter ANN Mid1 marks",min_value=0,max_value=30,step=1)
ann2=ann2.number_input("Enter ANN Mid2 marks",min_value=0,max_value=30,step=1)
if st.button("Predict"):
    inp=pd.Series([ml1,ml2,atcd1,atcd2,cn1,cn2,ipp1,ipp2,ann1,ann2])
    inp=inp.values.reshape((-1,10))
    predicted=ensemble.predict(inp)[0]
    predicted=round(predicted,2)
    progress=st.progress(0)
    for i in range(100):
        time.sleep(0.1)
        progress.progress(i+1)
    st.success("Hooray! Successfully Done!")
    st.header("Dear {} , Your Predicted Sgpa is {}    ".format(name,predicted))