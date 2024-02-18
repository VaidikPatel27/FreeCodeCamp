import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

def draw_plot():
    # Read data from file
    df = pd.read_csv('epa-sea-level.csv')

    # Create scatter plot
    plt.scatter(df['Year'],df['CSIRO Adjusted Sea Level'])
    plt.xlabel('Year')
    plt.ylabel('Sea Level (inches)')
    plt.title('Rise in Sea Level')

    # Create first line of best fit

    X = df['Year']
    y = df['CSIRO Adjusted Sea Level']
    X2 =  pd.DataFrame(range(1880,2051),columns=['year'])
    res = linregress(X,y)

    fig, (ax1, ax2) = plt.subplots(1, 2,figsize = (16,8))
    plt.subplot(1,2,1)
    ax1.plot(X,y,'o')
    ax1.plot(X2,res.intercept + res.slope*X2,linewidth=2,c='r')
    plt.xlabel('Year')
    plt.ylabel('Sea Level (inches)')
    plt.title('Rise in Sea Level')

    # Create second line of best fit
    new_df = df[df['Year']>=2000][['Year','CSIRO Adjusted Sea Level']]
    X_new = new_df['Year']
    y_new = new_df['CSIRO Adjusted Sea Level']
    X2_new =  pd.DataFrame(range(2000,2051),columns=['year'])
    res_new = linregress(X_new,y_new)
    plt.subplot(1,2,2)
    ax2.plot(X_new,y_new,'o')
    ax2.plot(X2_new,res_new.intercept + res_new.slope*X2_new,linewidth=2,c='r')
    plt.xlabel('Year')
    plt.ylabel('Sea Level (inches)')
    plt.title('Rise in Sea Level')
    
    
    

    # Add labels and title

    
    # Save plot and return data for testing (DO NOT MODIFY)
    plt.savefig('sea_level_plot.png')
    return plt.gca()