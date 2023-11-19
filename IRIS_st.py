import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.title('**IRIS PREDICTION APP 🌸**')
st.markdown("""
- *This app predict iris flower type* 
""")
st.sidebar.header(' input parameters')


def user_input_features():
    sepal_length = st.sidebar.slider('Sepal_length', 4.2, 7.2, 5.2)
    sepal_width = st.sidebar.slider('Sepal_width', 2.0, 4.4, 3.2)
    petal_length = st.sidebar.slider('Petal_length', 1.0, 6.9, 3.4)
    petal_width = st.sidebar.slider('Petal_width', 0.1, 2.5, 0.3)
    data ={
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }
    features = pd.DataFrame(data,index=[0])
    return features


df = user_input_features()
st.subheader('User Input Params')
st.write(df)
iris = datasets.load_iris()
x = iris.data
y = iris.target

clf = RandomForestClassifier()
clf.fit(x,y)
prediction = clf.predict(df)
prediction_prob = clf.predict_proba(df)
st.subheader("""
Class label and their corresponding index
""")
st.write('- iris target names')
st.subheader('PREDICTION')
st.write(prediction)
st.write(iris.target_names[prediction])
st.write(prediction)
st.subheader('Predictions Probability')
st.write(prediction_prob)
