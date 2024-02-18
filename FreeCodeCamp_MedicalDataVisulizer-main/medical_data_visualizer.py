import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
df['overweight'] = np.where((round(df['weight']/(df['height']/100)**2,1)) >= 25.0,1,0)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.


# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    cardio_columns = ['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke', 'cardio']
    # df_cat = \
    df_cardio_long = pd.melt(df, id_vars='cardio', value_vars=cardio_columns[:-1],value_name='value')


    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cardio_long = pd.DataFrame({'total':(df_cardio_long.groupby(['cardio','variable'])['value']).value_counts()}).reset_index()

    

    # Draw the catplot with 'sns.catplot()'
    catplot = sns.catplot(df_cardio_long,
                          x='variable',
                          y='total',
                          col='cardio',
                          hue='value',
                          kind='bar')


    # Get the figure for the output
    fig = catplot


    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df[(df['ap_lo']>=0) & (df['ap_hi']>=0)]
    df_heat = df_heat[df_heat['ap_lo']<=df_heat['ap_hi']]

    for col in df_heat[['height', 'weight']].columns:
        lower_percentile = df_heat[col].quantile(0.025)
        higher_percentile = df_heat[col].quantile(0.975)

        df_heat = df_heat[(df[col] >= lower_percentile) & (df_heat[col] <= higher_percentile)]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool_)
    mask[np.triu_indices_from(mask)] = True




    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(round(corr, 1),
                          annot=True,
                          mask=mask,
                          cmap='twilight_shifted',
                          linewidths=0.5)
    # Draw the heatmap with 'sns.heatmap()'
    fig = heatmap.figure


    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
