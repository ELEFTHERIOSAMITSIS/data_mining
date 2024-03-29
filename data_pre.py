import pandas as pd
import os



def create_dataframes():
    folder_path='harth/'
    files = os.listdir(folder_path)
    dfs = {}
    count_dfs=0
    for file in files:
        file_path = os.path.join(folder_path, file) 
        dfname=os.path.splitext(file)[0]
        dfs[dfname] = pd.read_csv(file_path)
        dfs[dfname].set_index('timestamp', inplace=True)

        count_dfs+=1
    return dfs



#def remove_columns():


