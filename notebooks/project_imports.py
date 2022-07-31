#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd 
from pathlib import Path
import seaborn as sns
import warnings


class DirectoryMapper:

    def __init__(self, project_name, base_dir = Path('/Users/kevinstoltz/Documents/Projects/Kaggle')):

        # Map folders I will likely use for any ML pipeline
        self.base_dir = base_dir
        self.project_dir = base_dir/project_name

        # Map to config file and load base section of config
        self.config_path = self.project_dir/'configs'

        # Map data folders (and SQL folder)
        self.sql_dir = self.project_dir/'sql'
        self.data_dir = self.project_dir/'data'
        self.raw_data_dir = self.data_dir/'raw'
        self.interim_data_dir = self.data_dir/'interim'
        self.processed_data_dir = self.data_dir/'processed'
        self.external_data_dir = self.data_dir/'external'

        # Map output folders (and model folder)
        self.model_dir = self.project_dir/'models'
        self.output_dir = self.project_dir/'reports'
        self.output_data_dir = self.output_dir/'data'
        self.output_figs_dir = self.output_dir/'figures'
        self.output_pred_dir = self.output_dir/'predictions'
        
        
    def load_data(self, file_name, level=None, file_type='parquet'):

            # If user uses .parquet or .csv, strip off the . 
            if file_type[0] == '.':
                file_type = file_type[1:]

            # Attach file name and file type extension
            file_name = f'{file_name}.{file_type}'

            # Load from full input path or lazy load from file name and data level
            if level is None:
                in_file_name=file_name 
            elif level == 'raw':
                in_file_name = self.raw_data_dir/file_name
            elif level == 'interim':
                in_file_name = self.interim_data_dir/file_name
            elif level == 'processed':
                in_file_name = self.processed_data_dir/file_name
            elif level == 'external':
                in_file_name = self.external_data_dir/file_name
            else:
                raise ValueError('Please provide valid level')

            # Different file type loads require different Pandas methods
            if file_type == 'parquet':
                return pd.read_parquet(in_file_name)
            elif file_type == 'csv':
                return pd.read_csv(in_file_name)
            else: 
                raise ValueError('Provide valid file type')
            
            
def handle_date_cols(df, date_cols, add_date_parts=True, **kwargs):
    
    '''
    Converts date columns that may have come in as strings to datetime type with an 
    option to create date parts out of those newly created dates for downstream input
    into a machine learning algorithm.

    Parameters
    ----------

    df: A Pandas DataFrame object
        A feature set, possibly with dates represented as strings.

    date_cols: list
        A list of date columns that are represented as strings (i.e. need to be converted
        to a datetime format).

    add_date_parts: boolean
        Option to convert each datetime field into date parts.

    Returns
    -------

    A Pandas DataFrame object with dates represented as datetime objects and possibly
    date parts for each of those datetime objects.
    '''
    
    for d in date_cols:
        df[d] = pd.to_datetime(df[d]) # They come in as strings so make sure to convert to datetime first
        
    if add_date_parts:
        df = create_date_parts(df, **kwargs)
        
    return df


def create_date_parts(df, drop_date_cols=True):
    
    '''
    Find all datetime type columns in a feature set and create year, month, week, and 
    day date parts for that oject. This transformation is required for passing dates
    into a machine learning model as most ML models require all data types to be float.

    Parameters
    ----------

    df: A Pandas DataFrame object.
        Feature set with datetime objects that need to be converted into their
        individual components. 

    Returns
    -------

    A Pandas DataFrame object with dates broken down into their individual elements.
    The original datetime objects remain in the DataFrame object as well.
    '''
    
    # Find columns that include data type 'datetime64' and return as a list
    date_cols = list(df.select_dtypes(include=['datetime64']).columns)
    
    for c in date_cols:
        for d in ['YEAR', 'MONTH']:
            if d == 'YEAR':
                df[f'{c}_{d}'] = df[f'{c}'].dt.year
            elif d == 'MONTH':
                df[f'{c}_{d}'] = df[f'{c}'].dt.month
                
    if drop_date_cols:
        df.drop(date_cols, inplace=True, axis=1)
    
    return df


def df_multiple_summary(df):
    
    print('info')
    print(df.info())
    
    print()
    
    print('unique')
    print(df.nunique())
    
    
    return df


def groupby_and_agg(df, groupby_cols, agg_, collapse_multicolumn=True, return_cols=None, **kwargs):
    
    '''
    Performs more complex group by and aggregation functions with automatic index reset and collapse of column
    names down to one level for operations that lead to multi-index ouputs.

    Parameters
    ----------

    df: Pandas DataFrame object
        A Pandas DataFrame containing at least one date column and one numerical column. 

    groupby_cols: str or list
        Column(s) to group by for downstream aggregation.
        
    agg_: str or dict
        A dictionary of df column names and their associated aggregations. All keys in agg_dict must match a column
        name in df argument.
        
    collapse_multicolumn: bool
        Collapses column names in multi-index column structures so that single column names are returned.
        
    returns_cols: list
        A subset of columns to return. All columns passed in groupby_cols argument will automaticall be added so 
        only include those additional columns that need to be returned. Default is to return all columns in df.
    
        
    Returns
    -------

    A Pandas DataFrame object containing a summary of data in original df argument. 

    '''
    
    # Right off the bat, let's check to make sure all keys in the agg_dict dictionary are columns in the df
    # Return original df if not
    if type(agg_) == dict:
        for a in agg_:
            if a not in df.columns:
                warnings.warn('Invalid column name passed in agg_ argument - returning original DataFrame')
                return df
            
    # Handle kwarg for index resetting - default to true
    if 'reset_index' in kwargs:
        assert type(kwargs['reset_index']) == bool 
        reset_index_ = kwargs['reset_index'] 
    else:
        reset_index_ = True
          
    # group by and aggregate
    df_agg = df.groupby(groupby_cols).agg(agg_)
    
    if reset_index_ == True:
        df_agg = df_agg.reset_index()

    # Handle multiple column indexes by collapsing the names down
    if collapse_multicolumn:
        if type(df_agg.columns) == pd.core.indexes.multi.MultiIndex:
            df_agg.columns = ['_'.join(col) for col in df_agg.columns]
            # I'll be left with columns that end in '_' so remove that character from ends
            df_agg.columns = [col.removesuffix("_") for col in df_agg.columns]
        else: 
            warnings.warn('No MultiIndex detected. No column collapse wil be performed')
            
    # Narrow down to specific return columns if needed - final state dependent on if index was reset
    if return_cols is None:
        return_cols = list(df_agg.columns)
    else:
        if reset_index_ == True:
            return_cols = groupby_cols + return_cols
     
    # Return df, narrowed down to specific columns if needed
    return df_agg[return_cols]




