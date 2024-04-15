import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt




def first_plots(test_df,test_df2=0):
    fig, axs = plt.subplots(2, 2, figsize=(50, 12))
    seconds_array_1 = np.arange(0,len(test_df)*0.002,0.002)
    
    axs[0,0].plot(seconds_array_1, test_df['back_x'], label='back_x')
    axs[0,0].plot(seconds_array_1, test_df['back_y'], label='back_y')
    axs[0,0].plot(seconds_array_1, test_df['back_z'], label='back_z')
    axs[0,0].set_ylabel('Back Sensor')
    axs[0,0].legend()

    axs[1,0].plot(seconds_array_1, test_df['thigh_x'], label='thigh_x')
    axs[1,0].plot(seconds_array_1, test_df['thigh_y'], label='thigh_y')
    axs[1,0].plot(seconds_array_1, test_df['thigh_z'], label='thigh_z')
    axs[1,0].set_ylabel('Thigh Sensor')
    axs[1,0].legend()
    seconds_array_2 = np.arange(0,len(test_df2)*0.002,0.002)
    axs[0,1].plot(seconds_array_2, test_df2['back_x'], label='back_x')
    axs[0,1].plot(seconds_array_2, test_df2['back_y'], label='back_y')
    axs[0,1].plot(seconds_array_2, test_df2['back_z'], label='back_z')
    axs[0,1].set_ylabel('Back Sensor')
    axs[0,1].legend()

    axs[1,1].plot(seconds_array_2, test_df2['thigh_x'], label='thigh_x')
    axs[1,1].plot(seconds_array_2, test_df2['thigh_y'], label='thigh_y')
    axs[1,1].plot(seconds_array_2, test_df2['thigh_z'], label='thigh_z')
    axs[1,1].set_ylabel('Thigh Sensor')
    axs[1,1].legend()

def plot_activity(activity, df,plotname,data_num):
    custom_x_values = range(0, data_num )
    data = df[df['label'] == activity][['back_x', 'back_y', 'back_z','thigh_x','thigh_y','thigh_z']][0:data_num]
    data.index = custom_x_values
    axis = data.plot(subplots=True, figsize=(16, 12), 
                     title=plotname)
    
    for ax in axis:
        ax.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5))
