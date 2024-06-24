import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1 Read .csv
df = pd.read_csv('medical_examination.csv')


# 2 setup overweight column
weight = df['weight']
height = df['height']
BMI = weight/((height/100)**2)
overweight = BMI > 25
# 2.1 Add the overweight column to the df variable
df['overweight'] = overweight.astype(int)
# 2.2 overweight conditional
# def is_overweight(bmi):
    # return 1 if bmi > 25 else 0
# 2.2.1 Apply the function to the BMI series and assign the result to the 'overweight' column
# df['overweight'] = BMI.apply(is_overweight)
print(df.head())


# 3 Setup cholestrol and gluc
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)


# 4 Setup catplot
def draw_cat_plot():
    # 5 define column
    cat_cols = ['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke']

    # 6 define cardio as splitter
    df_cat = pd.melt(df, id_vars=['Cardio'], value_vars=cat_cols, var_name='Category', value_name='Value')
    
    # 7 create catplot
    g = sns.catplot(x='Value', col='Category', hue='Cardio', kind='count', data=df_cat, palette='Set2')

    # 8 save and return the figure
    fig.savefig('catplot.png')
    return fig


# 10 setup heat map
def draw_heat_map():
    # 11 defining the heat map
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))

    # 12 define the correlation matrix
    corr = df_heat.corr()

    # 13 create mask
    mask = np.triu(np.ones_like(corr, dtype=bool))



    # 14 create heat map plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # 15 generate the heat map
    sns.heatmap(corr, annot=True, fmt='.1f', cmap='coolwarm', mask=mask, square=True, linewidths=.5, cbar_kws={"shrink": .5})


    # 16 save and return figure
    fig.savefig('heatmap.png')
    return fig
