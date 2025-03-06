#!/usr/bin/env python
# coding: utf-8

# #### Imports 

# In[1]:


from globals import *
from utils import *

import os
import re
import json
import pandas as pd
from collections import defaultdict

survey_df = pd.read_csv(survey_data)
annotations_dir = os.path.join(data_dir,'annotations')




# ##### Rank questions (where the answers are stored in one column) responses by demographic
# 

# In[ ]:


# Pick a question and demographic
question_col = "B.13.2"

demographics = ["Location","Role","Company Type","Dev Experience","Education", "Company Size", "Gender"]

demographic_key = "Location"

results = rank_question_answers_by_demographic(df=survey_df, question_col=question_col, demographic_key=demographic_key,
special_phrases=SPECIAL_PHRASES)

grouped_by_demo = results["grouped_by_demo"]
grouped_by_answer = results["grouped_by_answer"]

# Structured print
print("=== Ranking of answers BY DEMOGRAPHIC ===")
for group, answer_list in grouped_by_demo.items():
    print(f"\nDemographic Group: {group}")
    for ans, c, pct in answer_list:
        print(f"  {ans}: {c} mentions ({pct:.2f}%)")

print("\n=== Ranking of demographics BY ANSWER ===")
for ans, group_list in grouped_by_answer.items():
    print(f"\nAnswer: {ans}")
    for group, c, pct in group_list:
        print(f"  {group}: {c} mentions ({pct:.2f}%)")


# ### Functions to View Our Annotation Data 

# In[2]:


####################################################
# Generate theme and code distributions
# Search for texts associated with themes / codes 
####################################################

os.chdir(data_dir)

input_json_path =  'annotations/json/B_Definitions.json'   
print("=== Report by Question ===")
# Generates textual report, set output_graph = True to generate graph in figures/annotations
generate_annotator_report(input_json_path, output_graph=False)


input_json_path =  'annotations/json/A_Definitions.json'  
print("=== Report by Question ===")
generate_annotator_report(input_json_path, output_graph=False)

''' 
# Search for texts of a specific theme or code.
tag_query = "pns"  # Replace with your theme or code of interest
results = search_annotations_texts_by_tag(input_json_path, tag_query)

print(f"\n=== Responses with tag '{tag_query}' ===")
if results:
    for q_code, text in results:
        print(f"Question {q_code}: {text}")
else:
    print("No responses found with that tag.")
'''

# Rank annotations themes by Demographic 
results_b = analyze_demographic_themes('annotations/json/B_Definitions.json', 'Role', 'ai_study_finalized.csv')
# results_a = analyze_demographic_themes('json/A_Definitions.json', 'Role', '../ai_study_finalized.csv')

# The above results are also json that can be saved 
# print(json.dumps(results, indent=4))


# ##### Print graph and text for distribution of themes and codes found by role and principle in the written responses regarding risks and mitigation methods for AI ethics principles.

# In[ ]:


# Create graph displaying the distribution of themes and codes across different roles responses to how they view and mitigate risks to each principle of ethical AI

os.chdir(annotations_dir)

create_principle_role_graph('json/B_Risk_Mitigation.json', role_names, principle_names)

print_role_principle_analysis('json/B_Risk_Mitigation.json')

