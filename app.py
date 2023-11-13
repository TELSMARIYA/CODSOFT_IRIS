import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pickle
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
st.set_option('deprecation.showPyplotGlobalUse', False)



st.title('_IRIS FLOWER CLASSIFICATION_')
st.markdown("::white_flower::white_flower::white_flower::white_flower::white_flower:")

st.header('',divider='rainbow')
st.write('Do you want to know which category the iris flower belongs to? ')

st.markdown(''':black[IRIS Flower Classes in the dataset are:] :blue[_IRIS-SETOSA_,] :black[IRIS-VERSICOLOR,]
    :red[IRIS-VIRGINICA].''')

df=pd.read_csv(r"IRIS.csv")
if st.sidebar.checkbox('View Data', False):
    st.write(df)
if st.sidebar.checkbox('View Distributions', False):
    st.scatter_chart(data=df)
    plt.tight_layout()
    st.pyplot()
    


image = Image.open(r'iris-flower-meaning-and-symbolism.jpg')
st.image(image)

# step 1 : load the pickled model
model=open('rfc.pickle','rb')
clf=pickle.load(model)
model.close()
    
# step 2. get the front end user input 

sepal_length_=st.slider('sepal_length',4.2,8.0,4.2)
sepal_width_=st.slider('sepal_width',2.0,4.5,2.0)
petal_length_=st.slider('petal_length',1.0,7.0,1.0)
petal_width_=st.slider('petal_width',0.1,2.6,0.1)





sepal_length=(sepal_length_-df['sepal_length'].mean())/df['sepal_length'].std()
sepal_width=(sepal_width_-df['sepal_width'].mean())/df['sepal_width'].std()
petal_length=(petal_length_-df['petal_length'].mean())/df['petal_length'].std()
petal_width=(petal_width_-df['petal_width'].mean())/df['petal_width'].std()




#step 3: get the model input (convert user input to model impout)
input_data=[[sepal_length,sepal_width,petal_length,petal_width]]

#step 4: get the prediction and print the results 

prediction=clf.predict(input_data)[0]
if st.button('Predict'):
    if prediction==0:
        st.subheader('IRIS-SETOSA')
    elif prediction==1:
        st.subheader('IRIS-VERSICOLOR')
    elif prediction==2:
        st.subheader('IRIS-VIRGINICA')    
