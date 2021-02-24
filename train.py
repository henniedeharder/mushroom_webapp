import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
# from collections import defaultdict
import streamlit as st
from encoding import MultiColumnLabelEncoder

# modelling
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, auc, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

import xgboost as xgb 

import umap
import shap 
import pickle

# umap for plotting

def get_embedding(X):
    reducer = umap.UMAP()
    vals = X.values
    scaled = StandardScaler().fit_transform(vals)
    embedding = reducer.fit_transform(scaled)
    return embedding

def plot_embedding(embedding, X, y):
    plt.scatter(embedding[:, 0], embedding[:, 1], c=[sns.color_palette()[x] for x in y])
    plt.show()


# explain with shap
shap.initjs()

def get_shap(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values


def plot_summary(shap_values, X):
    shap.summary_plot(shap_values, X, max_display=120)
    plt.show()


# load and transform
def load_and_transform_mushroom_data():
    df = pd.read_csv('C:/git/RL/mushroom/mushrooms.csv')
    df.columns = [i.replace('-', ' ').replace('class', 'poisonous') for i in df.columns]

    trans_dict = {'poisonous':{'p':1, 'e':0}, 'cap shape':{'b':'bell','c':'conical','x':'convex','f':'flat','k':'knobbed','s':'sunken'}, 'cap surface':{'f':'fibrous','g':'grooves','y':'scaly','s':'smooth'}, 'cap color':{'n':'brown','b':'buff','c':'cinnamon','g':'gray','r':'green','p':'pink','u':'purple','e':'red','w':'white','y':'yellow'}, 'bruises':{'t':'yes','f':'no'}, 'odor':{'a':'almond','l':'anise','c':'creosote','y':'fishy','f':'foul','m':'musty','n':'none','p':'pungent','s':'spicy'}, 'gill attachment':{'a':'attached','d':'descending','f':'free','n':'notched'}, 'gill spacing':{'c':'close','w':'crowded','d':'distant'}, 'gill size':{'b':'broad','n':'narrow'}, 'gill color':{'k':'black','n':'brown','b':'buff','h':'chocolate','g':'gray','r':'green','o':'orange','p':'pink','u':'purple','e':'red','w':'white','y':'yellow'}, 'stalk shape':{'e':'enlarging', 't':'tapering'}, 'stalk root':{'b':'bulbous','c':'club','u':'cup','e':'equal','z':'rhizomorphs','r':'rooted','?':'missing'}, 'stalk surface above ring':{'f':'fibrous','y':'scaly','k':'silky','s':'smooth'}, 'stalk surface below ring':{'f':'fibrous','y':'scaly','k':'silky','s':'smooth'}, 'stalk color above ring':{'n':'brown','b':'buff','c':'cinnamon','g':'gray','o':'orange','p':'pink','e':'red','w':'white','y':'yellow'}, 'stalk color below ring':{'n':'brown','b':'buff','c':'cinnamon','g':'gray','o':'orange','p':'pink','e':'red','w':'white','y':'yellow'}, 'veil type':{'p':'partial','u':'universal'}, 'veil color':{'n':'brown','o':'orange','w':'white','y':'yellow'}, 'ring number':{'n':0,'o':1,'t':2}, 'ring type':{'c':'cobwebby','e':'evanescent','f':'flaring','l':'large','n':'none','p':'pendant','s':'sheathing','z':'zone'}, 'spore print color':{'k':'black','n':'brown','b':'buff','h':'chocolate','r':'green','o':'orange','u':'purple','w':'white','y':'yellow'}, 'population':{'a':'abundant','c':'clustered','n':'numerous','s':'scattered','v':'several','y':'solitary'}, 'habitat':{'g':'grasses','l':'leaves','m':'meadows','p':'paths','u':'urban','w':'waste','d':'woods'}}

    for col in df.columns:
        df[col] = df[col].map(trans_dict[col])

    cat = ['cap shape', 'cap surface', 'cap color', 'bruises', 'odor',
        'gill attachment', 'gill spacing', 'gill size', 'gill color',
        'stalk shape', 'stalk root', 'stalk surface above ring',
        'stalk surface below ring', 'stalk color above ring',
        'stalk color below ring', 'veil type', 'veil color',
        'ring type', 'ring number', 'spore print color', 'population', 'habitat']
    num = []

    cat_sel = ['odor', 'gill size', 'bruises', 'population', 'habitat', 'spore print color']
    cols = cat_sel + ['poisonous']

    enc = MultiColumnLabelEncoder(columns=cat_sel)

    le = enc.fit_transform(df)

    le = le[cols]
    return df, le, cat_sel, enc 


def train(df, target='poisonous', model=DecisionTreeClassifier(), display_results=True):
    y = df[target]
    X = df.drop(columns=target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cf = confusion_matrix(y_test, y_pred)

    if display_results:
        # display results
        print('Confusion matrix:')
        print(cf)
        print(f'Accuracy: {str(acc)}')

        shap = get_shap(model, X)
        plot_summary(shap, X)

        print('Embedding with UMAP')
        X_emb = get_embedding(X)
        plot_embedding(X_emb, X, y)

    return cf, acc, X, y, model


def load_model(model_filename):
    model = pickle.load(open(model_filename, 'rb'))
    return model

def predict(model, data, enc):
    encoded = enc.transform(data)
    pred = model.predict_proba(encoded)
    pred = pred.reshape((2,1))
    pred = pd.DataFrame(pred, index=['edible', 'poisonous'], columns=['score'])
    return pred


if __name__ == "__main__":
    # load and transform data
    df, le, cat_sel, enc = load_and_transform_mushroom_data()
    # train model
    cfle, accle, X_le, y_le, model_le = train(le, model=xgb.XGBClassifier(), display_results=False)
    # dump model
    pickle.dump(model_le, open('mushroom/mushroom.h5', 'wb'))