# import libraries
import sys


import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 50)

from sqlalchemy import create_engine

'''
The target of this module (process_data.py) is to write a data cleaning pipeline (ETL pipeline):

--Extract Data-- :
* Loads the messages and categories datasets
[disaster_categories.csv, disaster_messages.csv]

--Transform Data-- :
* Merges the two datasets
* Cleans the data

--Load Data-- :
* Stores it in a SQLite database
[SQLAlchemy engine, DATABASE name: DisasterResponse, and TableName:disaster_response]



'''

##### LOAD MESSAGES  AND CATEGORIES FROM CSV´S #####
def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories  data
    
    Arguments:
        messages_filepath -> Path to the   messages file
        categories_filepath -> Path to the  categories file
    Output:
        df -> Merge data  messages and data categories
    """

    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    #print(messages.head())
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
   # merge datasets messages and categories by id
    df = messages.merge(categories,on=('id'))
   
    return df
   

#####  CLEANING THE DATA To new Dataframe df  #####
def clean_data(df):
    
    """
    Cleaning data from new dataframe df
    
    Arguments:
        df -> Contains merged (data messages and categories)
        
    Output:
        df -> Contains clean data
    """

   
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";",expand=True)
    

    # select the first row of the categories dataframe
    row = categories.iloc[0]
    print('*cleaninig data categories*:3')

    # use this row to extract a list of new column names for categories.
    category_colnames = row.str.split("-").apply(lambda x: x[0])

    # rename the columns of `categories`
    categories.columns = category_colnames
    
    #Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        #name = categories[column].str.split("-")[1]
        categories[column] = categories[column].astype(str).str.split("-").apply(lambda x:x[1])
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    
    # drop the original categories column from `df`
    df = df.drop('categories',axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)

    # drop child_alone because all the values are 0
    df = df.drop('child_alone',axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df
    
   
#####    LOAD DATA IN DATABASE InsertDatabaseName, TABLE:disaster_response. #####
def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('disaster_response', engine, index=False,if_exists='replace')
 

def main():

    '''
To run ETL pipeline module that cleans data and stores in database

>>python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

- Origen Data
messages_filepath = data/disaster_messages.csv
categories_filepath = data/disaster_categories.csv

- Name DATABASE to Load cleaned data
database_filepath = data/DisasterResponse.db
    '''


    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        
#
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
