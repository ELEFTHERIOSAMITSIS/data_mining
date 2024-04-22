import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.api import qqplot 
import seaborn as sns

def plot_activity(activity, df,plotname,data_num):
    custom_x_values = range(0, data_num )
    data = df[df['label'] == activity][['back_x', 'back_y', 'back_z','thigh_x','thigh_y','thigh_z']][0:data_num]
    data.index = custom_x_values
    axis = data.plot(subplots=True, figsize=(16, 12), 
                     title=plotname)
    
    for ax in axis:
        ax.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5))

def plot_histograms_qqplots(df):
    colors = ['#FF4C4C', '#8CD790', '#4D7EA8', '#E97451',  '#F2B134']

# Create a 2x4 grid for histograms and Q-Q plots
    fig1 = plt.figure(figsize=(17, 3))
    fig1.set_facecolor('#131516')

    gs1 = gridspec.GridSpec(1, 6)

    ax1 = plt.subplot(gs1[0, 0])
    ax1.set_facecolor('whitesmoke')
    ax1.grid(True, linestyle='-', linewidth=0.8, color='lightgrey')
    plt.hist(df['back_x'], bins=100, color=colors[0], zorder=4)
    ax1.set_title('back_x', color='white')
    ax1.tick_params(axis='both', colors='white')  


    ax2 = plt.subplot(gs1[0, 1])
    ax2.set_facecolor('whitesmoke')
    ax2.grid(True, linestyle='-', linewidth=0.8, color='lightgrey')
    plt.hist(df['back_y'], bins=100, color=colors[1], zorder=4)
    ax2.set_title('back_y', color='white')
    ax2.tick_params(axis='both', colors='white')


    ax3 = plt.subplot(gs1[0, 2])
    ax3.set_facecolor('whitesmoke')
    ax3.grid(True, linestyle='-', linewidth=0.8, color='lightgrey')
    plt.hist(df['back_z'], bins=100, color=colors[2], zorder=4)
    ax3.set_title('back_z', color='white')
    ax3.tick_params(axis='both', colors='white')

    ax4 = plt.subplot(gs1[0, 3])
    ax4.set_facecolor('whitesmoke')
    ax4.grid(True, linestyle='-', linewidth=0.8, color='lightgrey')
    plt.hist(df['thigh_x'], bins=100, color=colors[3], zorder=4)
    ax4.set_title('thigh_x', color='white')
    ax4.tick_params(axis='both', colors='white')

    ax5 = plt.subplot(gs1[0, 4])
    ax5.set_facecolor('whitesmoke')
    ax5.grid(True, linestyle='-', linewidth=0.8, color='lightgrey')
    plt.hist(df['thigh_y'], bins=100, color=colors[4], zorder=4)
    ax5.set_title('thigh_y', color='white')
    ax5.tick_params(axis='both', colors='white')

    ax6 = plt.subplot(gs1[0, 5])
    ax6.set_facecolor('whitesmoke')
    ax6.grid(True, linestyle='-', linewidth=0.8, color='lightgrey')
    plt.hist(df['thigh_z'], bins=100, color=colors[4], zorder=4)
    ax6.set_title('thigh_z', color='white')
    ax6.tick_params(axis='both', colors='white')



    fig2 = plt.figure(figsize=(12, 3))
    fig2.set_facecolor('#131516')
    gs2 = gridspec.GridSpec(1, 6)

    # # QQ-Plots
    ax7 = plt.subplot(gs2[0, 0])
    ax7.set_facecolor('whitesmoke')
    ax7.grid(True, linestyle='-', linewidth=0.8, color='lightgrey')
    qqplot(df['back_x'], line='s', ax=ax7, color=colors[0], zorder=2)
    ax7.set_title('back_x QQ-Plot', color='white')
    ax7.tick_params(axis='both', colors='white')
    ax7.set_ylabel('Sample Quantiles', color='white')
    ax7.set_xlabel('Theoretical Quantiles', color='white')

    ax8 = plt.subplot(gs2[0, 1])
    ax8.set_facecolor('whitesmoke')
    ax8.grid(True, linestyle='-', linewidth=0.8, color='lightgrey')
    qqplot(df['back_y'], line='s', ax=ax8, color=colors[1], zorder=2)
    ax8.set_title('back_y QQ-Plot', color='white')
    ax8.tick_params(axis='both', colors='white')
    ax8.set_ylabel('Sample Quantiles', color='white')
    ax8.set_xlabel('Theoretical Quantiles', color='white')

    ax9 = plt.subplot(gs2[0, 2])
    ax9.set_facecolor('whitesmoke')
    ax9.grid(True, linestyle='-', linewidth=0.8, color='lightgrey')
    qqplot(df['back_z'], line='s', ax=ax9, color=colors[2], zorder=2)
    ax9.set_title('back_z QQ-Plot', color='white')
    ax9.tick_params(axis='both', colors='white')
    ax9.set_ylabel('Sample Quantiles', color='white')
    ax9.set_xlabel('Theoretical Quantiles', color='white')

    ax10 = plt.subplot(gs2[0, 3])
    ax10.set_facecolor('whitesmoke')
    ax10.grid(True, linestyle='-', linewidth=0.8, color='lightgrey')
    qqplot(df['thigh_x'], line='s', ax=ax10, color=colors[3], zorder=2)
    ax10.set_title('thigh_x QQ-Plot', color='white')
    ax10.tick_params(axis='both', colors='white')
    ax10.set_ylabel('Sample Quantiles', color='white')
    ax10.set_xlabel('Theoretical Quantiles', color='white')

    ax11 = plt.subplot(gs2[0, 4])
    ax11.set_facecolor('whitesmoke')
    ax11.grid(True, linestyle='-', linewidth=0.8, color='lightgrey')
    qqplot(df['thigh_y'], line='s', ax=ax11, color=colors[4], zorder=2)
    ax11.set_title('thigh_y QQ-Plot', color='white')
    ax11.tick_params(axis='both', colors='white')
    ax11.set_ylabel('Sample Quantiles', color='white')
    ax11.set_xlabel('Theoretical Quantiles', color='white')

    ax12 = plt.subplot(gs2[0, 5])
    ax12.set_facecolor('whitesmoke')
    ax12.grid(True, linestyle='-', linewidth=0.8, color='lightgrey')
    qqplot(df['thigh_z'], line='s', ax=ax12, color=colors[4], zorder=2)
    ax12.set_title('thigh_z QQ-Plot', color='white')
    ax12.tick_params(axis='both', colors='white')
    ax12.set_ylabel('Sample Quantiles', color='white')
    ax12.set_xlabel('Theoretical Quantiles', color='white')

    plt.tight_layout()
    plt.show()


def create_corr_matrix(df):
    cols=['back_x','back_y','back_z','thigh_x','thigh_y','thigh_z']
    df_plot=df[cols]
    plt.figure(figsize=(30,15))
    sns.heatmap(df_plot.corr().abs().round(2),annot=True, cmap = "BuGn")
    plt.show()

