import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error as MAE  


def process_data(df):
    """This fucntion is used to preprocess the dataframe and make the gallery and era categorical variables."""
    # Convert the start date and end date columns to datetime objects
    df['start date'] = pd.to_datetime(df['start date'], format='%d/%m/%Y')
    df['end date'] = pd.to_datetime(df['end date'], format='%d/%m/%Y')
    df['start_month'] = df['start date'].dt.month
    df['end_month'] =  df['end date'].dt.month
    df = df.dropna(subset= ['avg_month_search','total attendance','reach'])
    df = df.rename (columns= {'number of days':'num_days' , 'total attendance': 'total_attendance'})
    df = df[df['gallery'].isin(['Main', 'Sackler','GJW'])]
    df ['period'] = df['start date'].dt.to_period(freq='Q-AUG')
    df ['fy'] = df['period'].dt.qyear 

    
    #Era coded
    df.loc[df['fy'].isin ([2016,2017,2018,2019]),'era'] = 'Pre-covid'
    df.loc[df['fy'].isin ([2020,2021]),'era'] = 'Covid'
    df.loc[df['fy']>=2022,'era'] = 'Post-covid'

    
    df['gallery_type_encoded'] = df['gallery'].astype('category').cat.codes
    df['era_encoded'] = df['era'].astype('category').cat.codes
    df = df[~df.exhibition.str.contains('Emin|Summer Exhibition 2020')] #manually excluding some outliers - Emin/Munch and 2020 SE
    
    
    return df


def build_model_df(df,other=[]):
    attrs= ['exhibition','gallery_type_encoded','avg_month_search','total_attendance', 'reach','num_days','era_encoded']+other
    return df[attrs]

def log_transform(df,other=[]):
    """
    This function is used to log transform the continuous variables.
    """
    attrs = ["avg_month_search",'reach','total_attendance','num_days'] +other

    for i in attrs:
        new_col = 'log_'+i
        df.loc[:,new_col] = np.log1p(df[i])

    return df


def winsorization (df, other =[]):

    """winsorze the log_reach, log_avg_month_search and log_num_days.
    This will make the data points lower than 5% and higher than 95% to 5% and 95%.
    """
    attrs = ["log_avg_month_search",'log_reach','log_num_days'] +other

    for i in attrs:

        temp = i +"_ex_outlier"
        # Manual Winsorization
        low, high = df[i].quantile([0.05, 0.95])
        df[temp] = df[i].clip(lower=low, upper=high)
    return df


