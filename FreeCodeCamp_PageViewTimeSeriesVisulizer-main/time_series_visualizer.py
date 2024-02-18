import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Import data (Make sure to parse dates. Consider setting index column to 'date'.)
df = pd.read_csv('fcc-forum-pageviews.csv',index_col='date')

# Clean data
df.index = pd.DatetimeIndex(df.index)
df = df[df['value']<=df['value'].quantile(0.975)]
df = df[df['value']>=df['value'].quantile(0.025)]


def draw_line_plot():
    # Draw line plot
    plt.figure(figsize=(20, 6))
    plt.xlabel('Date')
    plt.ylabel('Page Views')
    plt.title('Daily freeCodeCamp Forum Page Views 5/2016-12/2019')
    fig = plt.plot(df.index,
                        df.value,
                        c='r')

    # Save image and return fig (don't change this part)
    fig.savefig('line_plot.png')
    return fig

def draw_bar_plot():
    # Copy and modify data for monthly bar plot
    df['ord'] = df.index.month
    df['day'] = df.index.day
    df['year'] = df.index.year
    df = df.sort_values('ord')
    df['month'] = df.index.month_name()
    df_bar = pd.DataFrame(df.groupby(['year', 'month', 'day', 'ord'])['value'].mean()).reset_index()
    df_bar = df_bar.sort_values('ord')




    # Draw bar plot
    plt.figure(figsize=(12, 14))
    fig = sns.barplot(x=df_bar['year'],
                          y=df_bar['value'],
                          hue=df_bar['month'],
                          errorbar=None,
                          palette='tab10')
    plt.ylabel('Average Page Views')
    plt.legend(loc='upper left', fontsize=17)




    # Save image and return fig (don't change this part)
    fig.savefig('bar_plot.png')
    return fig

def draw_box_plot():
    # Prepare data for box plots (this part is done!)
    df['ord'] = df.index.month
    df['day'] = df.index.day
    df['year'] = df.index.year
    df = df.sort_values('ord')
    df['month'] = df.index.month_name()
    df_box = pd.DataFrame(data.groupby(['year', 'month', 'day', 'ord'])['value'].mean()).reset_index()
    df_box = df_box.sort_values('ord')

    # Draw box plots (using Seaborn)

    fig, axs = plt.subplots(1, 2, figsize=(27, 9))
    plt.subplot(1, 2, 1)
    sns.boxplot(df_box, x='year', y='value', ax=axs[0])
    plt.ylabel('page value')
    plt.title("Year-wise Box Plot (Trend)")
    plt.subplot(1, 2, 2)
    sns.boxplot(df_box, x='month', y='value', ax=axs[1])
    plt.ylabel('page value')
    plt.title("Month-wise Box Plot (Seasonality)")


    # Save image and return fig (don't change this part)
    fig.savefig('box_plot.png')
    return fig
