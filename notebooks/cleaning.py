#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import os
from globals import *
os.chdir(data_dir)

def update_second_csv_with_location(first_csv, second_csv):
    """
    For each row in the second CSV, if its 'responseId' exists in the first CSV,
    this function appends the location (derived from the first CSV's filename without 
    the .csv extension) to the row's 'location' column. If the 'location' column does not 
    exist, it is created with an empty string. If the 'location' column already has a value, 
    the new location is appended (separated by a comma). Finally, the 'location' column is 
    moved to the 17th column position and the second CSV file is overwritten with the update.
    
    Parameters:
        first_csv (str): Path to the first CSV file. Its filename (minus the .csv extension)
                         is used as the location value.
        second_csv (str): Path to the second CSV file which will be updated.
    """
    # Read the CSV files
    df1 = pd.read_csv(first_csv)
    df2 = pd.read_csv(second_csv)
    
    # Extract the base filename (without .csv) to use as the location value
    base_name = os.path.basename(first_csv)
    location_value = base_name[:-4] if base_name.lower().endswith('.csv') else base_name
    
    # Create a set of responseIds from the first CSV for fast lookup
    response_ids = set(df1['ResponseId'])
    
    # Ensure 'location' column exists in df2; if not, create it with empty strings.
    # If it exists, fill NaN with empty string.
    if 'location' not in df2.columns:
        df2['location'] = ''
    else:
        df2['location'] = df2['location'].fillna('')
    
    # Update the location column: for matching responseIds, append the new location value
    def update_location(row):
        current = row['location']
        # Ensure current is a string (in case it was a float/NaN)
        if pd.isna(current):
            current = ''
        if row['ResponseId'] in response_ids:
            return current + (', ' if current else '') + location_value
        return current

    df2['location'] = df2.apply(update_location, axis=1)
    
    # Reinsert the 'location' column as the 17th column (index 16)
    location_series = df2.pop('location')
    insert_position = 16 if len(df2.columns) >= 16 else len(df2.columns)
    df2.insert(insert_position, 'location', location_series)
    
    # Overwrite the second CSV file with the updated DataFrame
    df2.to_csv(second_csv, index=False)



################
# NOTE: This is not getting some of the data
# Please fix 

update_second_csv_with_location('north america.csv', 'survey_finalized.csv')


# In[ ]:




