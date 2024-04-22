import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import scipy.stats as stats
from sklearn.decomposition import PCA
from scipy.stats import anderson
from scipy.stats import pearsonr
from hampel import hampel     ##rolling means 1st

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



def pca_test(df):
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(df)

def get_balanced_dataset(df,k=7000):
    balanced_data = pd.DataFrame()
    for i in range(1,13):
        activity = df[df['label']== i].head(k).copy()
        balanced_data=pd.concat([balanced_data, activity])
   
    print(balanced_data.shape)
    return balanced_data

def test_normality_anderson(df):
    cols=['back_x','back_y','back_z','thigh_x','thigh_y','thigh_z']
    print(f"{'DISTRIBUTION':<15}{'%sig':<10}{'stat':<12}{'crit-val':<10}"
      f"{'result':<10}\n")

# Loop through all continuous random variables and test them
    for var in cols:
        test = anderson(df[var])
        # Loop through test results and unpack the sig.levels and crit-vals
        for i in range(len(test.critical_values)):
            sig_lev, cv = test.significance_level[i], test.critical_values[i]
            # Check if test.stat is < crit-val
            result = 'Fail to reject' if test.statistic < cv else 'Reject'
            # Print results in tabular format
            print(f"{var:<15}{sig_lev:<10}{test.statistic:<12.3f}{cv:<10}"
                f"{result:<10}")  
            if i is 4:
                print('\n')

def correlation_analysis(df):
    cols=['back_x','back_y','back_z','thigh_x','thigh_y','thigh_z']
    # Set header for tabular output
    print(f"{'RANDOM VARIABLES':<25}{'corr':<10}{'p-value':<10}\n")
    # Iterate over continuous features list
    for i in range(len(cols)):
        # Set output variable
        y_output = cols[i]
        # Loop through continuous features list again
        for j in range(len(cols)):
            # Set input variable
            x_input = cols[j]
            # Check for equality, if equal, skip iteration
            if y_output == x_input:
                continue
            # If not equal, grab coefficient and p-value
            else:
                corr, p_val = pearsonr(df[x_input], df[y_output])
                
                print(f"{y_output + '-vs-' + x_input:<25}"
                    f"{corr:<10.2f}{p_val:<10.8f}")
            

def descritize_by_bounds(df):
    cols=['back_x','back_y','back_z','thigh_x','thigh_y','thigh_z']
    num_stdv = 1
# Create bounds for continuous labels
    for col in df.columns:
        if col in cols:
            col_mean = df[col].mean()
            col_stdv = df[col].std()
            lower_bound = col_mean - col_stdv * num_stdv
            upper_bound = col_mean + col_stdv * num_stdv
            bins = [-float('inf'), lower_bound - 0.5 * col_stdv, lower_bound + 0.5 * col_stdv, 
                upper_bound - 0.5 * col_stdv, upper_bound + 0.5 * col_stdv, float('inf')]
            df[col] = pd.cut(df[col], bins=bins, labels=['very low', 'low', 'avg', 'high', 'very high'])

    return df



def data_agg(df,chuck_size=100):
    total_chunks = len(df) // chuck_size

    # Initialize an empty DataFrame to store aggregated data
    aggregated_dfs = []
    # Iterate through each chunk
    for i in range(total_chunks):
        # Select the chunk of rows
        start_index = i * chuck_size
        end_index = (i + 1) * chuck_size
        df_chunk = df.iloc[start_index:end_index]

        # Calculate the mean for each column in the chunk
        chunk_mean = df_chunk.mean()
        # Append the chunk mean to the aggregated DataFrame
        aggregated_dfs.append(chunk_mean)
    df_aggregated = pd.concat(aggregated_dfs, axis=1).T

    return df_aggregated
