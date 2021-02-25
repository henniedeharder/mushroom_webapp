# run by running 'streamlit run streamlit_app.py'
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
st.write("""## Is this mushroom edible?  \nLet's find out! Specify the mushroom on the left.""")

# specify selectbox in sidebar
opts = []

for col in cat_sel:
    option = st.sidebar.selectbox(f'{col}', df[col].unique())
    opts.append(option)

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

if st.button('Send'):
    if pred.loc['edible','score'] > 0.5:
        prob = round(pred.loc['edible','score']*100)
        st.write(f"""This mushroom is **edible**!  \nEdibility score: **{str(prob)}**""")
        st.pyplot(fig)
    else:
        prob = round(pred.loc['edible','score']*100)
        st.write(f"""This mushroom is **poisonous**.  \nEdibility score: **{str(prob)}**""")
        st.pyplot(fig)
