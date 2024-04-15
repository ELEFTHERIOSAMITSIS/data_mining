import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import scipy.stats as stats

def create_dataframe():
    folder_path='harth/'
    files = os.listdir(folder_path)
    dfs = []
    for file in files:
        df = pd.read_csv( os.path.join(folder_path, file))
        participant_id = os.path.splitext(file)[0] 
        df['Participant_ID'] = participant_id
        if file=='S006':
            df['Timestamp'] = pd.to_datetime(df['Timestamp']) + pd.Timedelta(milliseconds=1)
        dfs.append(df)
    dataset=pd.concat(dfs)
    dataset.reset_index
    dataset.loc[dataset['label'] == 13, 'label'] = 9
    dataset.loc[dataset['label'] == 14, 'label'] = 10
    dataset.loc[dataset['label'] == 130, 'label'] = 11
    dataset.loc[dataset['label'] == 140, 'label'] = 12   
    return dataset


def window_data(time_step,overlap,df):

    step = time_step - overlap
    windows = []
    labels = []
    for i in range(0, len(df) - time_step, step):
        back_xs = df['back_x'].values[i: i + time_step]
        back_ys = df['back_y'].values[i: i + time_step]
        back_zs = df['back_z'].values[i: i + time_step]
        thigh_xs = df['thigh_x'].values[i: i + time_step]
        thigh_ys = df['thigh_y'].values[i: i + time_step]
        thigh_zs = df['thigh_z'].values[i: i + time_step]
        label = stats.mode(df['label'][i: i + time_step])[0]
        #print(f"{label} in lines --> {i}-{i+time_step}")
        windows.append([back_xs, back_ys, back_zs,thigh_xs,thigh_ys,thigh_zs])
        labels.append(label)
    return windows,labels    


def normal_data_Standard(df):
    scaler = StandardScaler()
    cols=['back_x','back_y','back_z','thigh_x','thigh_y','thigh_z']
    df[cols] = scaler.fit_transform(df[cols])
    return df

def normal_data_MinMax(df):
    scaler = MinMaxScaler()
    cols=['back_x','back_y','back_z','thigh_x','thigh_y','thigh_z']
    df[cols] = scaler.fit_transform(df[cols])
    return df


def rolling_means(df,k=20):
    df=df.rolling(k).mean()
    new_df=df[k-1:]
    return new_df



