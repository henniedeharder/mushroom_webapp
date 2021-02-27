# run by running 'streamlit run C:\git\RL\mushroom\mushroomapp.py' in anaconda terminal from environment mushroom
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from explain import get_explanations, create_interpret_df, plot_interpretation
from train import load_and_transform_mushroom_data, predict, load_model

# load and transform data
df, le, cat_sel, enc = load_and_transform_mushroom_data()

# app title
st.write("""## **MUSHROOM EDIBILITY METER**  """)
st.write("""Is your mushroom edible? Let's find out! Specify the mushroom on the left and press Send.""")

# create sidebar
# st.sidebar.markdown('Mushroom specs')

opts = []

for col in cat_sel:
    option = st.sidebar.selectbox(f'{col}', df[col].unique())
    opts.append(option)

# st.sidebar.button('Send')

# get selection options
new_mush = pd.DataFrame([opts], columns=cat_sel)

# get prediction
model = load_model(model_filename='C:/git/RL/mushroom/mushroom.h5')
pred = predict(model, new_mush, enc)

# show explanation
new = enc.transform(new_mush)
shap_df = get_explanations(new, model)
interpretation = create_interpret_df(shap_df, new, new_mush)
fig, ax = plot_interpretation(interpretation)

col1, col2 = st.beta_columns([0.1,1])

with col1:
    send = st.button('Send')

with col2:
    info = st.button('Info')

if send:
    if pred.loc['edible','score'] > 0.5:
        prob = round(pred.loc['edible','score']*100)
        st.write(f"""This mushroom is **edible**!  \nEdibility score: **{str(prob)}**""")
        st.pyplot(fig)
    else:
        prob = round(pred.loc['edible','score']*100)
        st.write(f"""This mushroom is **poisonous**.  \nEdibility score: **{str(prob)}**""")
        st.pyplot(fig)

if info:
    size = 128, 128
    mush = Image.open('C:/git/RL/mushroom/img/mush.jpg')
    mush.thumbnail(size, Image.ANTIALIAS)
    st.image(mush)

    # st.write('Hi there,')
    st.write('With this app you can test whether a mushroom is poisonous or edible.  \n You can specify the mushroom in the sidebar and press send.  \n The six characteristics are identified as most important in predicting the edibility of mushrooms.  \nThe model is validated and was 100 percent accurate.')  
    st.write('The explanations are from shap. Green specs contribute in a positive way to edibility, while red ones make the mushroom less edible.')  
    st.write('*Use at your own risk!*')
    st.write('&#169 Hennie de Harder, 2021')
    
    
