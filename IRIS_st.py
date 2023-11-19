import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.title('**IRIS PREDICTION APP 🌸**')
st.markdown("""
- *This app predict iris flower type* 
""")
st.subheader('Iris Genus')
st.markdown(""" - *Iris Genus:The Iris genus encompasses a diverse group of flowering plants known for their stunning and varied blooms.
Iris flowers are characterized by their unique petal patterns and come in a wide array of colors.
The genus includes both herbaceous plants and bulbous species. Irises are widely cultivated for their ornamental value
and are symbolic of beauty and connection *""")
st.subheader('Iris setosa')
st.markdown("""
- Iris setosa is one of the three species in the Iris genus. It is known for its distinctive appearance,
characterized by short, sturdy stems and showy flowers with petals that can be white, blue, or pink.
Habitat: Found in cold regions, including parts of North America and Europe.""")
st.subheader('Iris versicolor')
st.markdown("""
- Iris versicolor, also known as the "Harlequin Blue flag," displays a wide range of colors, including blue, 
violet, and occasionally pink. It is recognized for its striking appearance and can be found in wetland areas.
Habitat: Native to North America, particularly in wetlands and along the edges of ponds.""")
st.subheader('Iris virginica')
st.markdown("""
- Iris virginica, commonly known as the "Virginia Iris" or "Southern Blue Flag," 
features blue to violet flowers with distinctive veining on the petals.
It is a wetland-loving species and is recognized for its elegance.
Habitat: Native to the eastern United States and typically found in marshy areas.
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
st.subheader('Predictions Probability')
st.write(prediction_prob)
