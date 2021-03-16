import numpy as np 
import pandas as pd 
import shap
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches


def get_explanations(df, model):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df)
        shap_df = pd.DataFrame(shap_values[1], columns=df.columns)
        return shap_df
    except Exception as e:
        print(e)


def create_interpret_df(shap_df, new, new_mush):
    try:
        name = []
        shap_values = []
        feature_names = []
        for i in range(len(shap_df)):
            name += [new.index[i]] * 6
            shap_values += sorted(list(np.array(shap_df.iloc[i])), reverse=True)[:3] + sorted(list(np.array(shap_df.iloc[i])))[:3][::-1]
            feature_names += list(shap_df.columns[np.array(shap_df.iloc[i]).argsort()[::-1][:3]]) + list(shap_df.columns[np.array(shap_df.iloc[i]).argsort()[::][:3][::-1]])

        interpret = pd.DataFrame({'idx':name, 'shap_value':shap_values, 'feature':feature_names})
        interpret['shap_value'] = interpret['shap_value'] * -1
        interpret = interpret.sort_values(by='shap_value', ascending=False)

        def get_value(row):
            return new_mush[new_mush.index == row.idx][row['feature']].values[0]

        interpret['feature_value'] = interpret.apply(get_value, axis=1)
        interpret['feature&value'] = interpret.apply(lambda row: row.feature + ' is ' + str(row.feature_value), axis=1)
        interpret['color'] = np.where(interpret['shap_value'] > 0, 'green', 'red')
        
        return interpret
    except Exception as e:
        print(e)


def plot_interpretation(interpret):
    y_pos = np.arange(len(interpret.index))
    # plt.rcParams["figure.figsize"] = (10,10)
    fig, ax = plt.subplots(figsize=(8,3))
    ax.barh(y_pos, interpret.shap_value.values, align='center', color=interpret.color.values)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(interpret['feature&value'].values)
    # ax.set_ylabel('Characteristic')
    ax.invert_yaxis() 
    ax.set_title('Contribution to edibility')
    plt.tick_params(axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False)
    green = mpatches.Patch(color='green', label='Positive contribution')
    red = mpatches.Patch(color='red', label='Negative contribution')
    plt.legend(handles=[green, red])
    return fig, ax