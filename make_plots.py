import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#from statsmodels.api import qqplot 
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

def plot_activity(activity, df,data_num):
    custom_x_values = range(data_num, data_num*2 )
    dataplot = df[df['label'] == activity][['back_x', 'back_y', 'back_z','thigh_x','thigh_y','thigh_z']][0:data_num]
    dataplot.index = custom_x_values
    fig1, ax1 = plt.subplots(figsize=(16, 4))
    dataplot.iloc[:, :3].plot(ax=ax1, title='back')
    ax1.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    
    # Plot the remaining three features in another figure
    fig2, ax2 = plt.subplots(figsize=(16, 4))
    dataplot.iloc[:, 3:].plot(ax=ax2, title='thigh')
    ax2.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    
    plt.show()

def create_corr_matrix(df):
    cols=['back_x','back_y','back_z','thigh_x','thigh_y','thigh_z']
    df_plot=df[cols]
    plt.figure(figsize=(30,15))
    sns.heatmap(df_plot.corr().abs().round(2),annot=True, cmap = "BuGn")
    plt.show()

def cluster3D(data,labels):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='viridis')
        ax.set_title('HAR-Clustering in 3D')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        ax.legend(*scatter.legend_elements(), title="Clusters")
        plt.show()
def cluster2D(data,labels):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    # XY Plane
    axes[0].scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    axes[0].set_title('X-Y Plane')
    axes[0].set_xlabel('Principal Component 1')
    axes[0].set_ylabel('Principal Component 2')

    # XZ Plane
    axes[1].scatter(data[:, 0], data[:, 2], c=labels, cmap='viridis')
    axes[1].set_title('X-Z Plane')
    axes[1].set_xlabel('Principal Component 1')
    axes[1].set_ylabel('Principal Component 3')

    # YZ Plane
    axes[2].scatter(data[:, 1], data[:, 2], c=labels, cmap='viridis')
    axes[2].set_title('Y-Z Plane')
    axes[2].set_xlabel('Principal Component 2')
    axes[2].set_ylabel('Principal Component 3')

    plt.tight_layout()
    plt.show()