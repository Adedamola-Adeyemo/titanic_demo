import streamlit as st
import joblib
import pandas as pd

file = open('model.joblib','rb')
model = joblib.load(file)

st.title('Titanic Survival Prediction')
st.subheader('Which Passenger survived the shipwreck?')
st.write('Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.')

html_temp = """
    <div style ='background-color: orange; padding:10px'>
    <h2> Streamlit ML Web App </h2>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)

st.write('Please provide the following details for passenger')
pclass = st.selectbox('Pclass',('1','2','3'))
sex = st.selectbox('Sex - 0 for female, 1 for male',('0','1'))
age = st.slider('Age',0,80)
sibsp = st.selectbox('Sibsp',('0','1','2','3'))
parch = st.selectbox('Parch',('0','1','2','3'))
fare = st.slider('Fare',0,512,4)
dest = st.slider('Home.dest',0,500)
embarked_C = st.selectbox('Embarked_C',('0','1'))
embarked_Q = st.selectbox('Embarked_Q',('0','1'))
embarked_S = st.selectbox('Embarked_S',('0','1'))

features = {'pclass':pclass,
'sex':sex,
'age':age,
'sibsp':sibsp,
'parch':parch,
'fare':fare,
'home.dest':dest,
'embarked_C':embarked_C,
'embarked_Q':embarked_Q,
'embarked_S':embarked_S
}
if st.button('Submit'):
    data = pd.DataFrame(features,index=[0,1])
    st.write(data)

    prediction = model.predict(data)
    proba = model.predict_proba(data)[1]

    if prediction[0] == 0:
        st.error('The passenger died')
    else:
        st.success('The passenger survived')

    proba_df = pd.DataFrame(proba,columns=['Probability'],index=['Died','Survived'])
    proba_df.plot(kind='barh')
    st.pyplot()