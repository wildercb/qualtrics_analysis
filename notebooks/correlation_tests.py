#!/usr/bin/env python
# coding: utf-8

# ### Imports and Defenitions

# In[ ]:


import os 

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.stats import spearmanr, chi2_contingency
import matplotlib.pyplot as plt

from globals import *



# #####################################
# #### Is there a correlation between AI Managers Familiarity with ethical principles of AI and effectiveness of integrating principles? (B.9.1)/(H1a)
# #####################################

# In[ ]:


#######################################################
# Is there a correlation between AI Managers familiarity
# with principles and effectiveness of principle integration? (B.9.1) / (H1a)
#######################################################

# Define file path
file_path = survey_data

def read_and_clean_csv(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"Successfully read {file_path}.")
        return df
    except UnicodeDecodeError:
        print(f"Failed to read {file_path} with utf-8 encoding.")
        return None

# Read the CSV file
df = read_and_clean_csv(file_path)

if df is not None:
    # Define the column index for the effectiveness question and role column
    effectiveness_column = 'B.9.1'
    role_column_index = demographics['Role']['column']
    
    # Step 1: Filter data for AI Managers
    df['GroupedRole'] = df.iloc[:, role_column_index].map(demographics['Role']['mapping'])
    ai_manager_df = df[df['GroupedRole'] == 'AI Manager'].copy()

    # Step 2: Encode the effectiveness text responses safely
    le = LabelEncoder()  # Create an instance of LabelEncoder
    # Drop any NaNs in the effectiveness column before encoding
    ai_manager_df = ai_manager_df.dropna(subset=[effectiveness_column])
    ai_manager_df.loc[:, 'Effectiveness_Score'] = le.fit_transform(ai_manager_df[effectiveness_column].astype(str))

    # Step 3: Calculate familiarity scores for Tracks A and B
    familiarity_scores = []

    for track in ['A', 'B']:
        # Check that the column indices in principle_columns exist in the DataFrame
        valid_columns = [col for col in principle_columns[track] if col < ai_manager_df.shape[1]]
        valid_column_names = ai_manager_df.columns[valid_columns]

        # Calculate the familiarity score for each row by averaging all principle responses
        # NOTE THIS IS FOR AVERAGE FAMILIARITY OF ALL PRINCIPLES 
        track_familiarity_scores = ai_manager_df[valid_column_names].applymap(
            lambda level: familiarity_levels.index(level) + 1 if level in familiarity_levels else np.nan
        )
        avg_familiarity_scores = track_familiarity_scores.mean(axis=1, skipna=True)
        
        familiarity_scores.append(avg_familiarity_scores)

    # Combine familiarity scores across both tracks (A and B) into one series
    combined_familiarity_scores = pd.concat(familiarity_scores, axis=0)

    # Step 4: Align effectiveness scores to match combined familiarity scores and filter out NaNs
    aligned_effectiveness_scores = ai_manager_df['Effectiveness_Score'].reindex(combined_familiarity_scores.index)
    combined_data = pd.concat([combined_familiarity_scores, aligned_effectiveness_scores], axis=1).dropna()
    combined_familiarity_scores_cleaned = combined_data.iloc[:, 0]
    effectiveness_scores_cleaned = combined_data.iloc[:, 1]

    # Step 5: Compute correlation
    print(combined_familiarity_scores_cleaned, effectiveness_scores_cleaned)
    correlation, p_value = spearmanr(combined_familiarity_scores_cleaned, effectiveness_scores_cleaned)

    # Output the results
    print(f"Correlation between AI Manager familiarity and effectiveness (B.9.1): {correlation:.4f}")
    print(f"P-value: {p_value:.4f}")


# #######################################################
# #####  Is there a correlations between Requirements Analysts' familiarity with principles and:
# ##### 1. frequency of including ethics in documentation (B.10.1) / (H1b)
# ##### 2. number of principles considered in requirements (B.10.3) / (H1c)
# ##### 3. impact of Ethics on AI lifecycle (B.10.2)
# #######################################################

# In[ ]:


#######################################################
# Correlations between Requirements Analysts' familiarity
# with principles and:
# 1. frequency of including ethics in documentation (B.10.1) / (H1b)
# 2. number of principles considered in requirements (B.10.3) / (H1c)
#######################################################

# Read the CSV file
df = pd.read_csv(survey_data)
print("Successfully read AI_Study_Accepted.csv.")

# Filter for Requirements Analysts
role_column = df.columns[30]
req_analysts = df[df[role_column].isin(['Requirements Analyst or Engineer', 'Scrum Master, Product Manager, or Project Manager'])]
# Calculate average familiarity score for Track B (columns 79-87)
familiarity_columns = df.columns[79:88]
for col in familiarity_columns:
    req_analysts[col] = req_analysts[col].map(familiarity_mapping)
req_analysts['Avg_Familiarity'] = req_analysts[familiarity_columns].mean(axis=1)

# Get frequency scores (B.10.1)
frequency_column = 'B.10.1'
frequency_mapping = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3, 'Always': 4}
req_analysts['Frequency_Score'] = req_analysts[frequency_column].map(frequency_mapping)

# Count principles considered in requirements (B.10.3)
principles_column = 'B.10.3'
req_analysts['Principles_Count'] = req_analysts[principles_column].fillna('').apply(lambda x: len(x.split(',')) if x else 0)

# Prepare data for correlations
correlation_data = req_analysts[['Avg_Familiarity', 'Frequency_Score', 'Principles_Count']].dropna()

print(f"\nNumber of Requirements Analysts: {len(req_analysts)}")
print(f"Number of valid pairs for correlation: {len(correlation_data)}")

# Compute correlations if there's enough data
if len(correlation_data) > 1:
    corr_familiarity_frequency, p_value_freq = spearmanr(correlation_data['Avg_Familiarity'], correlation_data['Frequency_Score'])
    corr_familiarity_principles, p_value_princ = spearmanr(correlation_data['Avg_Familiarity'], correlation_data['Principles_Count'])
    
    print(f"\nCorrelation between familiarity and frequency (B.10.1): {corr_familiarity_frequency:.4f}")
    print(f"P-value: {p_value_freq:.4f}")
    
    print(f"\nCorrelation between familiarity and number of principles considered (B.10.3): {corr_familiarity_principles:.4f}")
    print(f"P-value: {p_value_princ:.4f}")
else:
    print("\nInsufficient data to compute correlations.")

# Print distributions
print("\nFamiliarity scores distribution:")
print(req_analysts['Avg_Familiarity'].value_counts(bins=5, sort=False))

print("\nFrequency scores distribution:")
print(req_analysts['Frequency_Score'].value_counts(sort=False))

print("\nNumber of principles considered distribution:")
print(req_analysts['Principles_Count'].value_counts(sort=False))

print("\nUnique responses in frequency column (B.10.1):")
print(req_analysts[frequency_column].unique())

print("\nSample of principles considered (B.10.3):")
print(req_analysts[principles_column].head())


# In[ ]:


#######################################################
# Is there a correlation between Requirements analyst familiarity
# with principles and impact of Ethics on AI lifecycle? (B.10.2)
#######################################################
file_path = survey_data

def read_and_clean_csv(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"Successfully read {file_path}.")
        return df
    except UnicodeDecodeError:
        print(f"Failed to read {file_path} with utf-8 encoding.")
        return None

# Read the CSV file
df = read_and_clean_csv(file_path)

if df is not None:
    # Define the column index for the impact question and role column
    impact_column = 'B.10.2'
    role_column_index = demographics['Role']['column']
    
    # Step 1: Filter data for "Requirements Analysts"
    df['GroupedRole'] = df.iloc[:, role_column_index].map(demographics['Role']['mapping'])
    req_analyst_df = df[df['GroupedRole'] == 'Requirements analyst'].copy()

    # Step 2: Drop rows where the impact response is "Prefer not to say"
    req_analyst_df = req_analyst_df[req_analyst_df[impact_column] != 'Prefer not to say']

    # Step 3: Encode the impact text responses safely
    le = LabelEncoder()  # Create an instance of LabelEncoder
    req_analyst_df.loc[:, 'Impact_Score'] = le.fit_transform(req_analyst_df[impact_column].astype(str))

    # Step 4: Calculate familiarity scores for Tracks A and B
    familiarity_scores = []

    for track in ['A', 'B']:
        # Check that the column indices in principle_columns exist in the DataFrame
        valid_columns = [col for col in principle_columns[track] if col < req_analyst_df.shape[1]]
        valid_column_names = req_analyst_df.columns[valid_columns]

        # Calculate the familiarity score for each row by averaging all principle responses
        track_familiarity_scores = req_analyst_df[valid_column_names].applymap(
            lambda level: familiarity_levels.index(level) + 1 if level in familiarity_levels else np.nan
        )
        avg_familiarity_scores = track_familiarity_scores.mean(axis=1, skipna=True)
        
        familiarity_scores.append(avg_familiarity_scores)

    # Combine familiarity scores across both tracks (A and B) into one series
    combined_familiarity_scores = pd.concat(familiarity_scores, axis=0)

    # Step 5: Align impact scores to match combined familiarity scores and filter out NaNs
    aligned_impact_scores = req_analyst_df['Impact_Score'].reindex(combined_familiarity_scores.index)
    combined_data = pd.concat([combined_familiarity_scores, aligned_impact_scores], axis=1).dropna()
    combined_familiarity_scores_cleaned = combined_data.iloc[:, 0]
    impact_scores_cleaned = combined_data.iloc[:, 1]

    # Step 6: Compute correlation
    correlation, p_value = spearmanr(combined_familiarity_scores_cleaned, impact_scores_cleaned)

    # Output the results
    print(f"Correlation between Requirements Analyst familiarity and project impact (B.10.2): {correlation:.4f}")
    print(f"P-value: {p_value:.4f}")


# #######################################################
# ##### Correlation between AI Developers' familiarity with principles
# ##### and importance of transparency & explainability (B.11.2) / H1d
# ##### and importance of ethics considerations? (B.11.3) / H1e
# #######################################################

# In[ ]:


#######################################################
# Correlation between AI Developers' familiarity with principles
# and importance of transparency & explainability? (B.11.2) / h1d
# and importance of ethics considerations? (B.11.3) / H1e
#######################################################
df = pd.read_csv(survey_data)

role_mapping = {
    'AI Engineer or Developer': 'AI Developer',
    '(Software) Developer, Designer, or Architect': 'AI Developer',
    'Data Scientist or Data Analyst': 'AI Developer'
}

role_column = df.columns[30]
ai_developers = df[df[role_column].isin(role_mapping.keys())].copy()

# -------------------------------------------
# Calculate average familiarity (Track B)
# -------------------------------------------
familiarity_columns = df.columns[79:88]
for col in familiarity_columns:
    ai_developers[col] = ai_developers[col].map(familiarity_mapping)
ai_developers['Avg_Familiarity'] = ai_developers[familiarity_columns].mean(axis=1)

# -------------------------------------------
# B.11.3 - Correlation: familiarity vs confidence (H1e)
# -------------------------------------------
confidence_column = 'B.11.3'
ai_developers['Confidence_Score'] = ai_developers[confidence_column].map(confidence_mapping)

correlation_data_113 = ai_developers[['Avg_Familiarity', 'Confidence_Score']].dropna()

if len(correlation_data_113) > 1:
    corr_113, p_113 = spearmanr(
        correlation_data_113['Avg_Familiarity'],
        correlation_data_113['Confidence_Score']
    )
    print(f"\n[B.11.3] Correlation between Avg_Familiarity and Confidence: {corr_113:.4f}")
    print(f"[B.11.3] P-value: {p_113:.4f}")
else:
    print("\n[B.11.3] Insufficient data to compute correlation.")

# -------------------------------------------
# B.11.2 - Correlation: familiarity vs importance (H1d)
# -------------------------------------------
importance_column = 'B.11.2'
ai_developers['Importance_Score'] = ai_developers[importance_column].map(importance_mapping)

correlation_data_112 = ai_developers[['Avg_Familiarity', 'Importance_Score']].dropna()

if len(correlation_data_112) > 1:
    corr_112, p_112 = spearmanr(
        correlation_data_112['Avg_Familiarity'],
        correlation_data_112['Importance_Score']
    )
    print(f"\n[B.11.2] Correlation between Avg_Familiarity and Importance: {corr_112:.4f}")
    print(f"[B.11.2] P-value: {p_112:.4f}")
else:
    print("\n[B.11.2] Insufficient data to compute correlation.")


print("\nFamiliarity scores distribution:")
print(ai_developers['Avg_Familiarity'].value_counts(bins=5, sort=False))

print("\nConfidence scores distribution (B.11.3):")
print(ai_developers['Confidence_Score'].value_counts(sort=False))

print("\nImportance scores distribution (B.11.2):")
print(ai_developers['Importance_Score'].value_counts(sort=False))


# #######################################################
# ##### Correlation between QA roles' familiarity with principles
# ##### and importance of ethical considerations (B.12.2)
# ##### and confidence in addressing ethical considerations (B.12.3)
# #######################################################
# 
# 

# In[ ]:


#######################################################
# Correlation between QA roles' familiarity with principles
# and importance of ethical considerations (B.12.2) / H1f
# and confidence in addressing ethical considerations (B.12.3) / H1g
#######################################################

# Read the CSV file
df = pd.read_csv(survey_data)

# Define QA role mapping
qa_role_mapping = {
    '(Software) Quality Assurance Engineer or Tester': 'QA'
}

role_column = df.columns[30]
qa_roles = df[df[role_column].isin(qa_role_mapping.keys())].copy()
print(f"Total QA roles found: {len(qa_roles)}")

# -------------------------------------------------------
# Calculate average familiarity score for Track B
# -------------------------------------------------------
familiarity_columns = df.columns[79:88]
for col in familiarity_columns:
    qa_roles[col] = qa_roles[col].map(familiarity_mapping)

qa_roles['Avg_Familiarity'] = qa_roles[familiarity_columns].mean(axis=1)

# -------------------------------------------------------
# B.12.2 - QA Familiarity vs. Importance of Ethical Considerations H1f
# -------------------------------------------------------
importance_column = 'B.12.2'  # Adjust if your column name differs
qa_roles['Importance_Score'] = qa_roles[importance_column].map(importance_mapping)

correlation_data_12_2 = qa_roles[['Avg_Familiarity', 'Importance_Score']].dropna()
print(f"\nNumber of valid pairs for B.12.2 correlation: {len(correlation_data_12_2)}")

if len(correlation_data_12_2) > 1:
    corr_12_2, p_12_2 = spearmanr(
        correlation_data_12_2['Avg_Familiarity'],
        correlation_data_12_2['Importance_Score']
    )
    print(f"Correlation (B.12.2): {corr_12_2:.4f} | P-value: {p_12_2:.4f}")
else:
    print("Insufficient data to compute correlation (B.12.2).")

# Distributions for B.12.2
print("\nFamiliarity scores distribution (QA roles):")
print(qa_roles['Avg_Familiarity'].value_counts(bins=5, sort=False))

print("\nImportance scores distribution (B.12.2):")
print(qa_roles['Importance_Score'].value_counts(sort=False))

# Crosstab for B.12.2
print("\nCrosstab of familiarity and importance (B.12.2):")
familiarity_bins_12_2 = pd.cut(
    correlation_data_12_2['Avg_Familiarity'],
    bins=5,
    labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
)
crosstab_12_2 = pd.crosstab(familiarity_bins_12_2, correlation_data_12_2['Importance_Score'])
print(crosstab_12_2)

# Print unique scores & original responses for B.12.2
print("\nUnique importance scores in the data (B.12.2):")
print(correlation_data_12_2['Importance_Score'].unique())

print("\nOriginal responses for B.12.2:")
print(qa_roles[importance_column].value_counts())

# -------------------------------------------------------
# B.12.3 - QA Familiarity vs. Confidence in Ethical Considerations (H1g)
# -------------------------------------------------------
confidence_column = 'B.12.3'  # Adjust if your column name differs
qa_roles['Confidence_Score'] = qa_roles[confidence_column].map(confidence_mapping)

correlation_data_12_3 = qa_roles[['Avg_Familiarity', 'Confidence_Score']].dropna()
print(f"\nNumber of valid pairs for B.12.3 correlation: {len(correlation_data_12_3)}")

if len(correlation_data_12_3) > 1:
    corr_12_3, p_12_3 = spearmanr(
        correlation_data_12_3['Avg_Familiarity'],
        correlation_data_12_3['Confidence_Score']
    )
    print(f"Correlation (B.12.3): {corr_12_3:.4f} | P-value: {p_12_3:.4f}")
else:
    print("Insufficient data to compute correlation (B.12.3).")

# Distributions for B.12.3
print("\nFamiliarity scores distribution (QA roles):")
print(qa_roles['Avg_Familiarity'].value_counts(bins=5, sort=False))

print("\nConfidence scores distribution (B.12.3):")
print(qa_roles['Confidence_Score'].value_counts(sort=False))


# Crosstab for B.12.3
print("\nCrosstab of familiarity and confidence (B.12.3):")
familiarity_bins_12_3 = pd.cut(
    correlation_data_12_3['Avg_Familiarity'],
    bins=5,
    labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
)
crosstab_12_3 = pd.crosstab(familiarity_bins_12_3, correlation_data_12_3['Confidence_Score'])
print(crosstab_12_3)

# Print unique scores & original responses for B.12.3
print("\nUnique confidence scores in the data (B.12.3):")
print(correlation_data_12_3['Confidence_Score'].unique())

print("\nOriginal responses for B.12.3:")
print(qa_roles[confidence_column].value_counts())


# ###########################################################################################
# #### Is there a correlation between (demographic) and familiarity with AI ethics principles? (H2)
# ###########################################################################################

# In[ ]:


##############################################
# 1. Read in the Data
##############################################
df = pd.read_csv(survey_data)

##############################################
# 3. Process Ethics Familiarity Columns by Track
##############################################
for track in principle_columns.keys():
    cols = principle_columns[track]
    # Ensure the column indices exist in the DataFrame
    valid_cols = [col for col in cols if col < df.shape[1]]
    # Get the actual column names from the indices
    col_names = [df.columns[col] for col in valid_cols]
    
    # Map each ethics response to its numeric value
    for col in col_names:
        df[col] = df[col].map(familiarity_mapping)
    
    # Compute the overall (average) ethics familiarity for this track
    avg_col_name = f'Avg_Ethics_Familiarity_{track}'
    df[avg_col_name] = df[col_names].mean(axis=1)
    print(f"Computed overall ethics familiarity for Track {track} in column '{avg_col_name}'.")

# For each demographic, create a "clean" (mapped) column and a numeric (factorized) column.
for demo_name, demo_info in demographics.items():
    col_index = demo_info['column']
    if col_index >= df.shape[1]:
        print(f"Warning: Column index {col_index} for {demo_name} not found in DataFrame.")
        continue
    col_name = df.columns[col_index]
    clean_col = f"{demo_name}_clean"
    num_col = f"{demo_name}_num"
    
    if demo_info['mapping'] is not None:
        df[clean_col] = df[col_name].map(demo_info['mapping'])
    else:
        df[clean_col] = df[col_name]
    
    df[num_col], _ = pd.factorize(df[clean_col])

##############################################
# 5. Initialize List to Store Significant Results
##############################################
significant_results = []

##############################################
# 6. Statistical Tests for Ethics Familiarity vs. Demographics
##############################################
for track in ['A', 'B']:
    avg_ethics_col = f'Avg_Ethics_Familiarity_{track}'
    print(f"\n=== Ethics Familiarity (Track {track}) vs. Demographics ===\n")
    
    # Overall Ethics Familiarity vs. Demographics
    for demo_name in demographics.keys():
        num_col = f"{demo_name}_num"
        clean_col = f"{demo_name}_clean"
        test_data = df[[avg_ethics_col, num_col]].dropna()
        if test_data.empty:
            print(f"Insufficient data for {demo_name} (overall ethics familiarity, Track {track}).")
            continue
        
        # Spearman correlation (overall ethics familiarity vs. demographic numeric)
        sp_corr, sp_p = spearmanr(test_data[avg_ethics_col], test_data[num_col])
        print(f"{demo_name} (Overall Ethics): Spearman r = {sp_corr:.4f} (p = {sp_p:.4f})")
        if sp_p < 0.05:
            significant_results.append({
                'Track': track,
                'Test': 'Overall Ethics vs Demographic',
                'Demographic': demo_name,
                'Metric': 'Spearman',
                'Statistic': sp_corr,
                'p_value': sp_p
            })
        
        # Chi-Square test for overall ethics familiarity (rounded) vs. categorical demographic
        contingency_table = pd.crosstab(df[clean_col], df[avg_ethics_col].round())
        if contingency_table.size > 0:
            chi2_stat, chi2_p, dof, expected = chi2_contingency(contingency_table)
            print(f"   Chi-Square Statistic = {chi2_stat:.4f} (p = {chi2_p:.4f})")
            if chi2_p < 0.05:
                significant_results.append({
                    'Track': track,
                    'Test': 'Overall Ethics vs Demographic',
                    'Demographic': demo_name,
                    'Metric': 'Chi-Square',
                    'Statistic': chi2_stat,
                    'p_value': chi2_p
                })
        else:
            print("   Insufficient data for Chi-Square test.")
        print()
    
    # Per-Principle Ethics Familiarity vs. Demographics
    valid_cols = [col for col in principle_columns[track] if col < df.shape[1]]
    col_names = [df.columns[col] for col in valid_cols]
    
    for i, col in enumerate(col_names):
        principle_name = principles[i] if i < len(principles) else f"Principle_{i}"
        print(f"--- {principle_name} (Track {track}) ---")
        for demo_name in demographics.keys():
            num_col = f"{demo_name}_num"
            clean_col = f"{demo_name}_clean"
            sub_data = df[[col, num_col]].dropna()
            if sub_data.empty:
                print(f"{demo_name}: Insufficient data for {principle_name}.")
                continue
            
            # Spearman correlation for this principle vs. demographic
            sp_corr, sp_p = spearmanr(sub_data[col], sub_data[num_col])
            print(f"{demo_name}: Spearman r = {sp_corr:.4f} (p = {sp_p:.4f})", end="; ")
            if sp_p < 0.05:
                significant_results.append({
                    'Track': track,
                    'Test': 'Per-Principle Ethics vs Demographic',
                    'Demographic': demo_name,
                    'Principle': principle_name,
                    'Metric': 'Spearman',
                    'Statistic': sp_corr,
                    'p_value': sp_p
                })
            
            # Chi-Square test for this principle (categorical) vs. demographic (categorical)
            contingency = pd.crosstab(df[clean_col], df[col])
            if contingency.size > 0:
                chi2_stat, chi2_p, dof, expected = chi2_contingency(contingency)
                print(f"Chi-Square = {chi2_stat:.4f} (p = {chi2_p:.4f})")
                if chi2_p < 0.05:
                    significant_results.append({
                        'Track': track,
                        'Test': 'Per-Principle Ethics vs Demographic',
                        'Demographic': demo_name,
                        'Principle': principle_name,
                        'Metric': 'Chi-Square',
                        'Statistic': chi2_stat,
                        'p_value': chi2_p
                    })
            else:
                print("Insufficient data for Chi-Square test.")
        print()

##############################################
# 7. Print All Significant Results (p < 0.05)
##############################################
print("\n=== Significant Results (p < 0.05) ===\n")
if not significant_results:
    print("No tests reached significance (p < 0.05).")
else:
    for result in significant_results:
        if result['Test'] == 'Overall Ethics vs Demographic':
            desc = (f"Track {result['Track']} - {result['Test']} ({result['Demographic']} - {result['Metric']}): "
                    f"Statistic = {result['Statistic']:.4f}, p = {result['p_value']:.4f}")
        else:
            desc = (f"Track {result['Track']} - {result['Test']} ({result['Demographic']}, Principle: {result.get('Principle','N/A')} - {result['Metric']}): "
                    f"Statistic = {result['Statistic']:.4f}, p = {result['p_value']:.4f}")
        print(desc)


# ###########################################################################################
# ### Is there a correlation between (demographic) and familiarity with AI Governance Initiatives? (H3)
# ###########################################################################################

# In[ ]:


############################################################
# 1. Read and Prepare the Data
############################################################
df = pd.read_csv(survey_data)

# Assume governance regulation familiarity columns are columns 220-228.
familiarity_column_indices = list(range(220, 229))  # adjust if necessary
valid_columns = [col for col in familiarity_column_indices if col < df.shape[1]]
print(f"Valid familiarity columns (by index): {valid_columns}")

# Convert indices to column names
valid_column_names = [df.columns[col] for col in valid_columns]
print(f"Valid familiarity column names: {valid_column_names}")

if len(valid_column_names) < len(regulation_names):
    regulation_names = regulation_names[:len(valid_column_names)]
elif len(valid_column_names) > len(regulation_names):
    regulation_names.extend(['Unnamed Regulation'] * (len(valid_column_names) - len(regulation_names)))

# Map the familiarity responses to numeric scores for each regulation column.
for col in valid_column_names:
    df[col] = df[col].map(familiarity_mapping)

# Calculate the average familiarity score across all governance regulation columns.
df['Avg_Regulation_Familiarity'] = df[valid_column_names].mean(axis=1)

# For each demographic, create a "clean" column (using mapping, if provided)
# and a numeric version (using pd.factorize) for correlation tests.
for demo_name, demo_info in demographics.items():
    col_index = demo_info['column']
    if col_index >= df.shape[1]:
        print(f"Warning: Column index {col_index} for {demo_name} not found!")
        continue
    col_name = df.columns[col_index]
    clean_col = f"{demo_name}_clean"
    num_col = f"{demo_name}_num"
    
    if demo_info['mapping'] is not None:
        df[clean_col] = df[col_name].map(demo_info['mapping'])
    else:
        df[clean_col] = df[col_name]
    
    df[num_col], _ = pd.factorize(df[clean_col])

############################################################
# 4. Overall Familiarity vs. Each Demographic
############################################################
print("\n=== Overall Avg Regulation Familiarity vs. Demographics ===\n")
for demo_name in demographics.keys():
    num_col = f"{demo_name}_num"
    clean_col = f"{demo_name}_clean"
    
    # Prepare data: drop any rows with missing values for the overall familiarity or the demographic.
    overall_data = df[['Avg_Regulation_Familiarity', num_col]].dropna()
    if overall_data.empty:
        print(f"Insufficient data for {demo_name}.")
        continue

    # Spearman correlation using numeric values
    overall_corr, overall_p = spearmanr(overall_data['Avg_Regulation_Familiarity'], overall_data[num_col])
    print(f"{demo_name}:")
    print(f"  Spearman correlation (Avg_Regulation_Familiarity vs. {demo_name}): {overall_corr:.4f} (p = {overall_p:.4f})")
    
    # For the Chi-Square test, create a contingency table. Here we round the overall familiarity 
    overall_contingency = pd.crosstab(df[clean_col], df['Avg_Regulation_Familiarity'].round())
    if overall_contingency.size > 0:
        chi2_stat, chi2_p, dof, expected = chi2_contingency(overall_contingency)
        print(f"  Chi-Square Statistic: {chi2_stat:.4f} (p = {chi2_p:.4f})")
    else:
        print("  Insufficient data for Chi-Square test.")
    print()

############################################################
# 5. Per-Regulation Familiarity vs. Each Demographic (Optional)
############################################################
print("\n=== Per-Regulation Familiarity vs. Demographics ===\n")
for demo_name in demographics.keys():
    num_col = f"{demo_name}_num"
    clean_col = f"{demo_name}_clean"
    print(f"\n--- Demographic: {demo_name} ---")
    
    for i, reg_col in enumerate(valid_column_names):
        reg_name = regulation_names[i]
        print(f"\nRegulation: {reg_name}")
        
        # Spearman correlation for the current regulation and the demographic.
        reg_data = df[[reg_col, num_col]].dropna()
        if not reg_data.empty:
            reg_corr, reg_p = spearmanr(reg_data[reg_col], reg_data[num_col])
            print(f"  Spearman correlation: {reg_corr:.4f} (p = {reg_p:.4f})")
        else:
            print("  Insufficient data for Spearman correlation.")
        
        # Chi-Square test for the current regulation using the categorical demographic.
        reg_contingency = pd.crosstab(df[clean_col], df[reg_col])
        if reg_contingency.size > 0:
            chi2_reg, chi2_reg_p, dof, expected = chi2_contingency(reg_contingency)
            print(f"  Chi-Square Statistic: {chi2_reg:.4f} (p = {chi2_reg_p:.4f})")
        else:
            print("  Insufficient data for Chi-Square test.")


# In[ ]:


#######################################################
# Correlation between QA roles' familiarity with principles
# and importance of ethical considerations (B.12.2)
#######################################################

# Read the CSV file
df = pd.read_csv(survey_data)

# Define QA role mapping
qa_role_mapping = {
    '(Software) Quality Assurance Engineer or Tester': 'QA'
}

# Filter for QA roles
role_column = df.columns[30]
qa_roles = df[df[role_column].isin(qa_role_mapping.keys())].copy()

# Calculate average familiarity score for Track B (columns 79-87)
familiarity_columns = df.columns[79:88]
for col in familiarity_columns:
    qa_roles[col] = qa_roles[col].map(familiarity_mapping)
qa_roles['Avg_Familiarity'] = qa_roles[familiarity_columns].mean(axis=1)

# Map importance scores (B.12.2)
importance_column = 'B.12.2'
qa_roles['Importance_Score'] = qa_roles[importance_column].map(importance_mapping)

# Prepare data for correlation
correlation_data = qa_roles[['Avg_Familiarity', 'Importance_Score']].dropna()

print(f"\nNumber of QA roles: {len(qa_roles)}")
print(f"Number of valid pairs for correlation: {len(correlation_data)}")

# Compute correlation if there's enough data
if len(correlation_data) > 1:
    correlation, p_value = spearmanr(correlation_data['Avg_Familiarity'], correlation_data['Importance_Score'])
    print(f"\nCorrelation between QA roles' familiarity and importance of ethical considerations (B.12.2): {correlation:.4f}")
    print(f"P-value: {p_value:.4f}")
else:
    print("\nInsufficient data to compute correlation.")

# Print distributions
print("\nFamiliarity scores distribution:")
print(qa_roles['Avg_Familiarity'].value_counts(bins=5, sort=False))

print("\nImportance scores distribution:")
print(qa_roles['Importance_Score'].value_counts(sort=False))

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(correlation_data['Avg_Familiarity'], correlation_data['Importance_Score'])
plt.title("QA Roles: Familiarity vs Importance of Ethical Considerations")
plt.xlabel("Average Familiarity with Principles")
plt.ylabel("Importance of Ethical Considerations")
plt.show()

# Optional: Display a crosstab of familiarity and importance scores
print("\nCrosstab of familiarity and importance scores:")
familiarity_bins = pd.cut(correlation_data['Avg_Familiarity'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
crosstab = pd.crosstab(familiarity_bins, correlation_data['Importance_Score'])
print(crosstab)

# Print unique importance scores
print("\nUnique importance scores in the data:")
print(correlation_data['Importance_Score'].unique())

# Print original responses for B.12.2
print("\nOriginal responses for B.12.2:")
print(qa_roles[importance_column].value_counts())


# In[ ]:


#######################################################
# Chi-Square Test between location (column 28) 
# and effectiveness in AI ethics integration (B.9.1)
#######################################################

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Read the CSV file (assuming it's named 'AI_Study_Accepted.csv')
df = pd.read_csv(survey_data)

location_column = df.columns[28]

# Map locations into groups
df['Grouped_Location'] = df[location_column].map(location_mapping)

# Use the B.9.1 column for effectiveness (replace with actual column name if it's different)
effectiveness_column = 'B.9.1'  # Replace with the actual column name for B.9.1
df['Effectiveness_Level'] = df[effectiveness_column].map(effectiveness_mapping)

# Drop rows with missing data for effectiveness or location
correlation_data = df[['Effectiveness_Level', 'Grouped_Location']].dropna()

# Print data for debugging
print("Unique values in 'Grouped_Location':")
print(correlation_data['Grouped_Location'].unique())

print("\nValue counts for 'Effectiveness_Level':")
print(correlation_data['Effectiveness_Level'].value_counts())

# Perform the Chi-Square Test of Independence
if correlation_data.empty:
    print("No valid data available for Chi-Square test.")
else:
    #######################
    # Chi-Square Test
    #######################
    # Create a contingency table (cross-tabulation of the two variables)
    contingency_table = pd.crosstab(correlation_data['Grouped_Location'], correlation_data['Effectiveness_Level'])

    # Perform Chi-Square test
    chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)

    # Display the Chi-Square results
    print("\nContingency Table:")
    print(contingency_table)

    print(f"\nChi-Square Statistic: {chi2_stat:.4f}")
    print(f"Degrees of Freedom: {dof}")
    print(f"P-Value: {p_val:.4f}")

    if p_val < 0.05:
        print("\nResult: There is a statistically significant association between location and effectiveness of AI ethics integration.")
    else:
        print("\nResult: There is no statistically significant association between location and effectiveness of AI ethics integration.")


# In[ ]:


#######################################################
# Chi-Square and Spearman correlation for location (column 28)
# and the selected principles from B.2.2 (Individual principle)
#######################################################

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, spearmanr

# Read the CSV file (assuming it's named 'AI_Study_Accepted.csv')
df = pd.read_csv('AI_Study_Accepted.csv')
print("Successfully read AI_Study_Accepted.csv.")

# Define the principles in B.2.2
principles = [
    'Respect for Human Rights', 'Data Protection and Right to Privacy',
    'Harm Prevention and Beneficence', 'Non-Discrimination and Freedom of Privileges',
    'Fairness and Justice', 'Transparency and Explainability of AI Systems',
    'Accountability and Responsibility', 'Democracy and Rule of Law',
    'Environment and Social Responsibility'
]

# Define location grouping (use column 28)
location_mapping = {
    'North America': 'North America',
    'EU/UK/EEA': 'Europe',
    'Europe - Outside of EU/UK/EEA': 'Europe',
    'Central/South America': 'Others',
    'Africa': 'Others',
    'Middle East': 'Others',
    'Asia': 'Others',
    'Australia and Oceania': 'Others'
}

# Use the correct location column (column 28, zero-indexed so it's 27)
location_column = df.columns[28]  # Adjusted for zero-indexed column

# Map locations into groups
df['Grouped_Location'] = df[location_column].map(location_mapping)

# Process the B.2.2 column for principles selection (replace 'B.2.2' with actual column name)
principles_column = 'B.2.2'  # Replace with the actual column name for B.2.2
df[principles_column] = df[principles_column].fillna('')

# Create binary columns for each principle
for principle in principles:
    df[principle] = df[principles_column].apply(lambda x: 1 if (principle in x or 'All' in x) else 0)

# Drop rows with missing location data
correlation_data = df[['Grouped_Location'] + principles].dropna()

# Print data for debugging
print("Unique values in 'Grouped_Location':")
print(correlation_data['Grouped_Location'].unique())

# Perform both Chi-Square and Spearman correlation for each principle
for principle in principles:
    print(f"\n### Analyzing principle: {principle} ###")

    #######################
    # Chi-Square Test
    #######################
    contingency_table = pd.crosstab(correlation_data['Grouped_Location'], correlation_data[principle])

    # Perform Chi-Square test
    chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)

    print("\nContingency Table:")
    print(contingency_table)

    print(f"\nChi-Square Statistic: {chi2_stat:.4f}")
    print(f"Degrees of Freedom: {dof}")
    print(f"P-Value: {p_val:.4f}")

    if p_val < 0.05:
        print(f"\nResult: There is a statistically significant association between location and selection of '{principle}'.")
    else:
        print(f"\nResult: There is no statistically significant association between location and selection of '{principle}'.")

    #######################
    # Spearman Correlation
    #######################
    # Convert Grouped_Location to numeric categories for Spearman correlation
    correlation_data['Location_Code'] = correlation_data['Grouped_Location'].astype('category').cat.codes

    # Perform Spearman correlation between principle selection and location code
    spearman_corr, spearman_p = spearmanr(correlation_data[principle], correlation_data['Location_Code'])

    # Display Spearman correlation results
    print(f"\nSpearman correlation for '{principle}': {spearman_corr:.4f}")
    print(f"P-value: {spearman_p:.4f}")


# In[ ]:


#######################################################
# Correlation between location (column 28)
# and familiarity with AI governance regulations(individual)
# Includes both Spearman and Chi-Square tests 
#######################################################
# Read the CSV file
df = pd.read_csv(survey_data)


# Define familiarity mapping for regulations
familiarity_mapping = {
    'Extremely Familiar': 5,
    'Moderately Familiar': 4,
    'Somewhat Familiar': 3,
    'Slightly Familiar': 2,
    'Not Familiar At All': 1
}

# Group locations
def group_location(location):
    if location in ['EU/UK/EEA', 'Europe - Outside of EU/UK/EEA']:
        return 'Europe'
    elif location == 'North America':
        return 'North America'
    else:
        return 'Others'

# Apply location grouping
location_column = df.columns[28]  # Adjust column index if necessary
df['Location_Group'] = df[location_column].apply(group_location)

# Columns related to AI governance regulations (e.g., columns 220-228)
familiarity_column_indices = list(range(220, 229))  # Check columns 220 to 228

# Ensure that the columns actually exist in the DataFrame
valid_columns = [col for col in familiarity_column_indices if col < df.shape[1]]
print(f"Valid columns are: {valid_columns}")

# Convert valid_columns (which are indices) into actual column names
valid_column_names = [df.columns[col] for col in valid_columns]
print(f"Valid column names are: {valid_column_names}")


# Truncate or pad the regulation_names list to match the number of valid columns
if len(valid_column_names) < len(regulation_names):
    regulation_names = regulation_names[:len(valid_column_names)]
elif len(valid_column_names) > len(regulation_names):
    regulation_names.extend(['Unnamed Regulation'] * (len(valid_column_names) - len(regulation_names)))

# Map familiarity levels to numerical scores for each valid column
for col in valid_column_names:
    df[col] = df[col].map(familiarity_mapping)

# Calculate average familiarity score across all valid governance initiative columns
df['Avg_Regulation_Familiarity'] = df[valid_column_names].mean(axis=1)

# Map location groups to numerical values
location_mapping = {'Europe': 0, 'North America': 1, 'Others': 2}
df['Location_Numeric'] = df['Location_Group'].map(location_mapping)

# Prepare data for correlation
correlation_data = df[['Avg_Regulation_Familiarity', 'Location_Numeric']].dropna()
print(f"\nNumber of valid pairs for correlation: {len(correlation_data)}")

# Compute Spearman correlation for overall familiarity
correlation, p_value = spearmanr(correlation_data['Avg_Regulation_Familiarity'], correlation_data['Location_Numeric'])
print(f"\nSpearman correlation between location and overall familiarity with AI governance regulations: {correlation:.4f}")
print(f"P-value: {p_value:.4f}")

# Perform both Spearman and Chi-Square tests for each regulation
print("\n--- Spearman Correlation and Chi-Square Test for each regulation ---\n")

for i, col in enumerate(valid_column_names):
    print(f"\nAnalyzing regulation: {regulation_names[i]}")

    #######################
    # Spearman Correlation
    #######################
    reg_correlation_data = df[[col, 'Location_Numeric']].dropna()
    
    if not reg_correlation_data.empty:
        spearman_corr, spearman_p = spearmanr(reg_correlation_data[col], reg_correlation_data['Location_Numeric'])
        print(f"Spearman correlation: {spearman_corr:.4f}")
        print(f"P-value: {spearman_p:.4f}")
    else:
        print("Insufficient data for Spearman correlation.")

    #######################
    # Chi-Square Test
    #######################
    # Create a contingency table for Chi-Square test
    contingency_table = pd.crosstab(df['Location_Group'], df[col])

    if contingency_table.size > 0:
        chi2_stat, chi_p_val, dof, expected = chi2_contingency(contingency_table)
        print(f"Chi-Square Statistic: {chi2_stat:.4f}")
        print(f"P-value: {chi_p_val:.4f}")
    else:
        print("Insufficient data for Chi-Square test.")


# In[ ]:


######################################
# Is there a correlation between primary applications and principles considere in development?
######################################
import pandas as pd
from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv(survey_data)

# Define columns for applications and ethics principles (assuming they are named appropriately)
application_column = "B.1.2"  # Column with all applications selected
principle_column = "B.2.2"    # Column with all principles selected

# Create an empty list to store results
results = []

# Step 1: Expand the applications column
df_app_expanded = df[application_column].str.split(',').explode().reset_index(drop=True)
df_principle_expanded = df[principle_column].str.split(',').explode().reset_index(drop=True)

# Combine exploded data into a new DataFrame
combined_df = pd.DataFrame({
    "Application": df_app_expanded.str.strip(),  # Strip spaces around values
    "Principle": df_principle_expanded.str.strip()  # Strip spaces around values
}).dropna()  # Drop rows where either Application or Principle is NaN

# Step 2: Identify unique applications and principles
unique_applications = combined_df["Application"].unique()
unique_principles = combined_df["Principle"].unique()

# Step 3: Run chi-square tests for each application-principle pair
for app in unique_applications:
    for principle in unique_principles:
        # Filter for the current application and principle to create a contingency table
        contingency_table = pd.crosstab(
            combined_df["Application"] == app,
            combined_df["Principle"] == principle
        )
        
        # Only run chi-square if we have a valid table (2x2 or greater)
        if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
            chi2, p, _, _ = chi2_contingency(contingency_table)
            results.append({
                "Application": app,
                "Principle": principle,
                "Chi2": chi2,
                "p-value": p
            })
            print(f"Chi-Square Test for {app} and {principle}")
            print(f"Chi2: {chi2:.4f}, p-value: {p:.4f}")
            if p < 0.05:
                print("Significant association found!")
            else:
                print("No significant association.")
        else:
            print(f"Insufficient data for Chi-Square test on '{app}' and '{principle}'.")

# Convert results to DataFrame
result_df = pd.DataFrame(results)

# Filter for significant results only
significant_results = result_df[result_df["p-value"] < 0.05]

# Step 4: Create a pivot table for visualization
pivot_table = significant_results.pivot(index="Application", columns="Principle", values="Chi2")

# Step 5: Plot heatmap for significant results
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt=".2f", cbar=True)
plt.title("Significant Associations Between AI Applications and Ethics Principles (Chi2)")
plt.show()


# In[ ]:


############################
# Is there a correlation between ethical principles considered and applications of AI chosen 
###############################


import pandas as pd
from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt
import re

# Load the data with proper handling for quoted commas
df = pd.read_csv("AI_Study_Accepted.csv", quotechar='"')
print("Successfully loaded data without skipping rows.")

# Define the column names upfront based on the dataset
application_column = "B.1.2"  # Primary applications of AI
principle_column = "B.2.2"    # Ethics principles considered

# Drop any non-data rows (adjust according to your actual CSV structure)
df = df.drop([0, 1]).reset_index(drop=True)

# Verify that the columns are still accessible
if application_column not in df.columns or principle_column not in df.columns:
    raise KeyError(f"Columns '{application_column}' or '{principle_column}' not found in the data.")

# List of known application options to match exactly
known_applications = [
    "Chatbots, Personal Assistants or Recommender Systems",
    "Computer Vision",
    "Customer Service",
    "Cybersecurity",
    "Entertainment or Communication",
    "Financial Services",
    "Healthcare",
    "Human Resources",
    "Legal",
    "Logistics",
    "Personalization and Advertisement",
    "Predictive Analysis",
    "Programming Analysis (e.g., Code Completion or Code Generation)",
    "Retail",
    "Robotics and Automation",
    "Translation or Text Generation",
    "Prefer not to say",
    "Other, please explain"
]

# Compile a regex pattern to match any of these known application options exactly
pattern = r'\b(?:' + '|'.join(re.escape(option) for option in known_applications) + r')\b'

# Step 1: Extract applications by finding exact matches for known applications
df_app_expanded = df[application_column].str.findall(pattern).explode().reset_index(drop=True)

# Step 2: Expand the principles by splitting on commas
df_principle_expanded = df[principle_column].str.split(',').explode().reset_index(drop=True)

# Step 3: Combine the expanded columns into a single DataFrame for analysis
combined_df = pd.DataFrame({
    "Application": df_app_expanded.str.strip(),  # Strip any extra whitespace
    "Principle": df_principle_expanded.str.strip()
}).dropna()  # Remove any rows where either Application or Principle is NaN

# Step 4: Get unique applications and principles
unique_applications = combined_df["Application"].unique()
unique_principles = combined_df["Principle"].unique()

# Step 5: Run chi-square tests for each application-principle pair
results = []
for app in unique_applications:
    for principle in unique_principles:
        # Create contingency table for chi-square test
        contingency_table = pd.crosstab(
            combined_df["Application"] == app,
            combined_df["Principle"] == principle
        )
        
        # Only run chi-square if the table is at least 2x2
        if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
            chi2, p, _, _ = chi2_contingency(contingency_table)
            results.append({
                "Application": app,
                "Principle": principle,
                "Chi2": chi2,
                "p-value": p
            })
            print(f"Chi-Square Test for {app} and {principle}")
            print(f"Chi2: {chi2:.4f}, p-value: {p:.4f}")
            if p < 0.05:
                print("Significant association found!")
            else:
                print("No significant association.")
        else:
            print(f"Insufficient data for Chi-Square test on '{app}' and '{principle}'.")

# Convert results to a DataFrame for visualization
result_df = pd.DataFrame(results)

# Filter for significant results
significant_results = result_df[result_df["p-value"] < 0.05]

# Create a pivot table and heatmap for significant results
pivot_table = significant_results.pivot(index="Application", columns="Principle", values="Chi2")

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt=".2f", cbar=True)
plt.title("Significant Associations Between AI Applications and Ethics Principles (Chi2)")
plt.show()


# In[ ]:


#####################
# Correlation between ethical principles seen at risk and applications of AI 
######################


import pandas as pd
from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt
import re

# Load the data with proper handling for quoted commas
df = pd.read_csv("AI_Study_Accepted.csv", quotechar='"')
print("Successfully loaded data without skipping rows.")

# Define the column names upfront based on the dataset
application_column = "B.1.2"  # Primary applications of AI
at_risk_principle_column = "B.2.4"  # Principles seen at risk

# Drop any non-data rows (adjust according to your actual CSV structure)
df = df.drop([0, 1]).reset_index(drop=True)

# Verify that the columns are still accessible
if application_column not in df.columns or at_risk_principle_column not in df.columns:
    raise KeyError(f"Columns '{application_column}' or '{at_risk_principle_column}' not found in the data.")

# List of known application options to match exactly
known_applications = [
    "Chatbots, Personal Assistants or Recommender Systems",
    "Computer Vision",
    "Customer Service",
    "Cybersecurity",
    "Entertainment or Communication",
    "Financial Services",
    "Healthcare",
    "Human Resources",
    "Legal",
    "Logistics",
    "Personalization and Advertisement",
    "Predictive Analysis",
    "Programming Analysis (e.g., Code Completion or Code Generation)",
    "Retail",
    "Robotics and Automation",
    "Translation or Text Generation",
    "Prefer not to say",
    "Other, please explain"
]

# Compile a regex pattern to match any of these known application options exactly
pattern = r'\b(?:' + '|'.join(re.escape(option) for option in known_applications) + r')\b'

# Step 1: Extract applications by finding exact matches for known applications
df_app_expanded = df[application_column].str.findall(pattern).explode().reset_index(drop=True)

# Step 2: Expand the principles seen at risk by splitting on commas
df_at_risk_principle_expanded = df[at_risk_principle_column].str.split(',').explode().reset_index(drop=True)

# Step 3: Combine the expanded columns into a single DataFrame for analysis
combined_df = pd.DataFrame({
    "Application": df_app_expanded.str.strip(),  # Strip any extra whitespace
    "At_Risk_Principle": df_at_risk_principle_expanded.str.strip()
}).dropna()  # Remove any rows where either Application or At_Risk_Principle is NaN

# Step 4: Get unique applications and at-risk principles
unique_applications = combined_df["Application"].unique()
unique_at_risk_principles = combined_df["At_Risk_Principle"].unique()

# Step 5: Run chi-square tests for each application-at-risk principle pair
results = []
for app in unique_applications:
    for principle in unique_at_risk_principles:
        # Create contingency table for chi-square test
        contingency_table = pd.crosstab(
            combined_df["Application"] == app,
            combined_df["At_Risk_Principle"] == principle
        )
        
        # Only run chi-square if the table is at least 2x2
        if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
            chi2, p, _, _ = chi2_contingency(contingency_table)
            results.append({
                "Application": app,
                "At_Risk_Principle": principle,
                "Chi2": chi2,
                "p-value": p
            })
            print(f"Chi-Square Test for {app} and {principle}")
            print(f"Chi2: {chi2:.4f}, p-value: {p:.4f}")
            if p < 0.05:
                print("Significant association found!")
            else:
                print("No significant association.")
        else:
            print(f"Insufficient data for Chi-Square test on '{app}' and '{principle}'.")

# Convert results to a DataFrame for visualization
result_df = pd.DataFrame(results)

# Filter for significant results
significant_results = result_df[result_df["p-value"] < 0.05]

# Create a pivot table and heatmap for significant results
pivot_table = significant_results.pivot(index="Application", columns="At_Risk_Principle", values="Chi2")

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt=".2f", cbar=True)
plt.title("Significant Associations Between AI Applications and At-Risk Principles (Chi2)")
plt.show()

