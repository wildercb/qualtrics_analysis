#!/usr/bin/env python
# coding: utf-8

# ### Imports and defenitions

# In[ ]:


import os 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re  # for filename sanitization
import matplotlib.patches as mpatches

from globals import survey_data, principle_columns, regulation_columns, principles, regulation_names, familiarity_levels, demographics

# Define file path
file_path = survey_data

colors = ['#FFFF99', '#FFEDA0', '#C7E9B4', '#7FCDBB', '#2C7FB8']



# ### A and B.2 Familiarity with Ethics Principles 
# 

# In[ ]:


def read_and_clean_csv(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"Successfully read {file_path}.")
        return df
    except UnicodeDecodeError:
        print(f"Failed to read {file_path} with utf-8 encoding.")
        return None

# ---------------- Chart Creation Functions ---------------- #

def create_chart(data, category, demographic, track):
    """Creates and saves an individual horizontal stacked bar chart."""
    if data.empty:
        print(f"No data available for {category} in {demographic} - Track {track}")
        return

    fig, ax = plt.subplots(figsize=(14, 4))
    y_pos = np.arange(len(data.index)) * 1.2
    bar_height = 0.8
    cumulative = np.zeros(len(data.index))

    for i, level in enumerate(familiarity_levels):
        values = data[level].values
        ax.barh(y_pos, values, left=cumulative, height=bar_height, label=level, color=colors[i])
        cumulative += values

    ax.set_xlabel('', fontweight='bold', fontname='Times New Roman')
    ax.set_title(f'Familiarity with AI Ethics Principles - {demographic}: {category} - Track {track}',
                 fontweight='bold', fontsize=16, fontname='Times New Roman')
    ax.set_xlim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for i, container in enumerate(ax.containers):
        text_color = 'black' if i < 3 else 'white'
        ax.bar_label(container, label_type='center', fmt='%.1f%%', fontname='Times New Roman',
                     fontweight='bold', fontsize=10, padding=2, color=text_color)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(data.index, fontweight='bold', fontname='Times New Roman')
    ax.set_ylim(y_pos.min() - bar_height/2, y_pos.max() + bar_height/2)

    legend_patches = [mpatches.Patch(color=color, label=level) for color, level in zip(colors, familiarity_levels)]
    fig.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.12), 
               ncol=5, fontsize=8, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    
    # Precompute safe strings to avoid f-string backslash issues
    safe_demographic = re.sub(r'[<>:"/\\|?*]', '_', demographic)
    safe_category = re.sub(r'[<>:"/\\|?*]', '_', category)
    plt.savefig(
        f'figures/familiarity_analysis/principle_familiarity/{track}.2.1/{track}.2.1_{safe_demographic}_{safe_category}_Track_{track}.png',
        dpi=300, bbox_inches='tight'
    )
    plt.close()


def create_combined_chart(all_data, demographic, track, graphs_per_row=3):
    """
    Creates and saves a combined chart that arranges multiple subplots in a grid.
    The number of graphs per row can be controlled with the graphs_per_row parameter.
    """
    num_categories = len(all_data)
    if num_categories == 0:
        print(f"No categories to plot for {demographic} - Track {track}")
        return

    # Determine number of rows needed
    num_rows = (num_categories + graphs_per_row - 1) // graphs_per_row
    total_axes = num_rows * graphs_per_row

    fig, axes = plt.subplots(num_rows, graphs_per_row, 
                             figsize=(10 * graphs_per_row, 4 * num_rows), 
                             sharey=False)
    # Ensure axes is always a 2D array
    if num_rows == 1:
        axes = np.array([axes])
    if graphs_per_row == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(f'Familiarity with AI Ethics Principles by {demographic} - Track {track}',
                 fontweight='bold', fontsize=16, fontname='Times New Roman')

    category_items = list(all_data.items())
    for idx, (category, data) in enumerate(category_items):
        row = idx // graphs_per_row
        col = idx % graphs_per_row
        ax = axes[row, col]

        y_pos = np.arange(len(data.index)) * 1.2
        bar_height = 0.8
        cumulative = np.zeros(len(data.index))

        for i, level in enumerate(familiarity_levels):
            values = data[level].values
            ax.barh(y_pos, values, left=cumulative, height=bar_height, label=level, color=colors[i])
            cumulative += values

        ax.set_xlabel('', fontweight='bold', fontname='Times New Roman', fontsize=10)
        ax.set_title(category, fontweight='bold', fontsize=12, fontname='Times New Roman')
        ax.set_xlim(0, 100)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for i, container in enumerate(ax.containers):
            text_color = 'black' if i < 3 else 'white'
            ax.bar_label(container, label_type='center', fmt='%.1f%%', fontname='Times New Roman',
                         fontweight='bold', fontsize=8, padding=2, color=text_color)

        # Only show y-axis tick labels on the first column to reduce clutter
        if col == 0:
            ax.set_yticks(y_pos)
            ax.set_yticklabels(data.index, fontweight='bold', fontname='Times New Roman', fontsize=8)
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])

        ax.set_ylim(y_pos.min() - bar_height/2, y_pos.max() + bar_height/2)
        for text in ax.get_xticklabels():
            text.set_fontname('Times New Roman')
            text.set_fontsize(8)

    # Hide any unused subplots
    for idx in range(num_categories, total_axes):
        row = idx // graphs_per_row
        col = idx % graphs_per_row
        fig.delaxes(axes[row, col])

    legend_patches = [mpatches.Patch(color=color, label=level) for color, level in zip(colors, familiarity_levels)]
    fig.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.05),
               ncol=5, fontsize=8, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, wspace=0.3)
    
    safe_demographic = re.sub(r'[<>:"/\\|?*]', '_', demographic)
    plt.savefig(
        f'figures/familiarity_analysis/principle_familiarity/{track}.2.1/{track}.2.1_{safe_demographic}_combined_Track_{track}.png',
        dpi=300, bbox_inches='tight'
    )
    plt.close()


# ---------------- Data Processing and Statistics ---------------- #

def process_and_create_charts(df, demographic, track):
    """
    Processes the DataFrame to group responses by a demographic category,
    generates individual charts for each group, and returns a dictionary of the data.
    """
    all_data = {}

    if demographic == 'Gender':
        categories = df.iloc[:, demographics[demographic]['column']].dropna().unique()
        df['GroupedCategory'] = df.iloc[:, demographics[demographic]['column']]
    else:
        categories = set(demographics[demographic]['mapping'].values())
        df['GroupedCategory'] = df.iloc[:, demographics[demographic]['column']].map(demographics[demographic]['mapping'])

    for category in categories:
        df_category = df[df['GroupedCategory'] == category]
        data = pd.DataFrame(index=principles, columns=familiarity_levels)
        valid_columns = principle_columns[track]
        df_track = df_category.dropna(subset=[df_category.columns[col] for col in valid_columns])
        total_respondents = len(df_track)

        for idx, principle in enumerate(principles):
            if idx < len(valid_columns):
                column = valid_columns[idx]
                if column < df_track.shape[1]:
                    counts = df_track.iloc[:, column].value_counts()
                    for level, count in counts.items():
                        if level in familiarity_levels:
                            data.loc[principle, level] = (count / total_respondents) * 100
        data = data.fillna(0)
        data = data[familiarity_levels]
        all_data[category] = data

        create_chart(data, category, demographic, track)

    return all_data


def print_stats(all_data, demographic, track):
    """
    Prints summary statistics for the processed data.
    """
    stats_summary = f"Statistics for {demographic} - Track {track}:\n\n"
    for category, data in all_data.items():
        stats_summary += f"{category}:\n"
        total_respondents = data.iloc[0].sum() / 100  
        for principle in principles:
            stats_summary += f"{principle}:\n"
            for level in familiarity_levels:
                percentage = data.loc[principle, level]
                count = round(percentage * total_respondents / 100)
                stats_summary += f" {level}: {count:.0f} ({percentage:.1f}%),"
            stats_summary += "\n"
        stats_summary += f"Total respondents: {total_respondents:.0f}\n"
        stats_summary += "\n" + "="*50 + "\n"
    
    print(stats_summary)


def process_combined_tracks(df, track):
    """
    Processes data for all participants in a single track (A or B) and creates a combined chart.
    """
    all_data = {}
    data = pd.DataFrame(index=principles, columns=familiarity_levels)
    valid_columns = principle_columns[track]
    df_track = df.dropna(subset=[df.columns[col] for col in valid_columns])
    total_respondents = len(df_track)

    for idx, principle in enumerate(principles):
        if idx < len(valid_columns):
            column = valid_columns[idx]
            if column < df_track.shape[1]:
                counts = df_track.iloc[:, column].value_counts()
                for level, count in counts.items():
                    if level in familiarity_levels:
                        data.loc[principle, level] = (count / total_respondents) * 100
                    else:
                        print(f"Unexpected familiarity level '{level}' for {principle}")
    data = data.fillna(0)
    data = data[familiarity_levels]
    all_data['All Participants'] = data

    create_combined_chart(all_data, 'All Participants', track, graphs_per_row=3)
    return all_data


def process_combined_all_tracks(df):
    """
    Processes data across both Track A and Track B and creates a combined chart.
    """
    all_data = {}
    combined_data = pd.DataFrame(index=principles, columns=familiarity_levels)
    total_respondents = 0

    # Accumulate total respondents across both tracks
    for track in ['A', 'B']:
        valid_columns = principle_columns[track]
        df_track = df.dropna(subset=[df.columns[col] for col in valid_columns])
        track_respondents = len(df_track)
        total_respondents += track_respondents

        for idx, principle in enumerate(principles):
            if idx < len(valid_columns):
                column = valid_columns[idx]
                if column < df_track.shape[1]:
                    counts = df_track.iloc[:, column].value_counts()
                    for level, count in counts.items():
                        if level in familiarity_levels:
                            current = combined_data.loc[principle, level]
                            if pd.isna(current):
                                current = 0
                            combined_data.loc[principle, level] = current + (count / total_respondents) * 100
                        else:
                            print(f"Unexpected familiarity level '{level}' for {principle}")
    combined_data = combined_data.fillna(0)
    combined_data = combined_data[familiarity_levels]
    all_data['All Participants'] = combined_data

    create_combined_chart(all_data, 'All Participants', 'Combined', graphs_per_row=3)
    return all_data


# ---------------- Main Execution ---------------- #

df = read_and_clean_csv(file_path)
if df is not None:
    for track in ['A', 'B']:
        print(f"Processing Track {track}...")
        track_columns = principle_columns[track]
        df_track = df.dropna(subset=[df.columns[col] for col in track_columns])
        for demographic in demographics:
            print(f"Processing {demographic}...")
            all_data = process_and_create_charts(df_track, demographic, track)
            create_combined_chart(all_data, demographic, track, graphs_per_row=3)
            print_stats(all_data, demographic, track)
        combined_track_data = process_combined_tracks(df, track)
    
    combined_all_tracks_data = process_combined_all_tracks(df)


# ### C Familiarity with Ethics Initiatives 
# 

# In[ ]:


def read_and_clean_csv(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"Successfully read {file_path}.")
        return df
    except UnicodeDecodeError:
        print(f"Failed to read {file_path} with utf-8 encoding.")
        return None

def create_chart(data, category, demographic):
    if data.empty:
        print(f"No data available for {category} in {demographic}")
        return

    fig, ax = plt.subplots(figsize=(18, 6))  # Increase figure size for larger output

    y_pos = np.arange(len(data.index)) * 1.2  # Increase spacing between bars
    bar_height = 0.8  # Bar height

    cumulative = np.zeros(len(data.index))

    for i, level in enumerate(familiarity_levels):
        values = data[level].values
        ax.barh(y_pos, values, left=cumulative, height=bar_height, label=level, color=colors[i])
        cumulative += values

    ax.set_xlabel('', fontweight='bold', fontname='Times New Roman')
    ax.set_title(f'Familiarity with AI Governance Initiatives - {demographic}: {category}',
                 fontweight='bold', fontsize=22, fontname='Times New Roman', pad=20)
    ax.set_xlim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for i, container in enumerate(ax.containers):
        text_color = 'black' if i < 3 else 'white'
        ax.bar_label(container, label_type='center', fmt='%.1f%%', fontname='Times New Roman',
                     fontweight='bold', fontsize=12, padding=4, color=text_color)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(data.index, fontweight='bold', fontname='Times New Roman', fontsize=14)
    ax.set_ylim(y_pos.min() - bar_height/2, y_pos.max() + bar_height/2)

    legend_patches = [mpatches.Patch(color=color, label=level) for color, level in zip(colors, familiarity_levels)]
    fig.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.15),
               ncol=5, fontsize=12, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)

    # Save the chart as an image file
    safe_category = re.sub(r'[<>:"/\\|?*]', '_', category)
    safe_demographic = re.sub(r'[<>:"/\\|?*]', '_', demographic)
    plt.savefig(f'figures/familiarity_analysis/governance_familiarity/C.2_{safe_demographic}_{safe_category}.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_combined_chart(all_data, demographic, graphs_per_row=3, categories_to_exclude=None):
    if categories_to_exclude is None:
        categories_to_exclude = []

    filtered_data = {k: v for k, v in all_data.items() if k not in categories_to_exclude}
    num_categories = len(filtered_data)

    if num_categories == 0:
        print(f"No categories to plot for {demographic} after excluding {categories_to_exclude}")
        return

    num_rows = (num_categories + graphs_per_row - 1) // graphs_per_row
    num_categories_adjusted = num_rows * graphs_per_row if num_categories % graphs_per_row != 0 else num_categories

    fig, axes = plt.subplots(num_rows, graphs_per_row, figsize=(8 * graphs_per_row, 6 * num_rows), sharey=False)
    fig.suptitle(f'Familiarity with AI Governance Initiatives - {demographic}', 
                 fontweight='bold', fontsize=22, fontname='Times New Roman', y=0.98 if num_rows > 1 else 1.05)

    category_items = list(filtered_data.items())
    for idx, (category, data) in enumerate(category_items):
        if idx >= num_categories_adjusted:
            break

        row = idx // graphs_per_row
        col = idx % graphs_per_row
        ax = axes[row, col] if num_rows > 1 else (axes[col] if graphs_per_row > 1 else axes)

        y_pos = np.arange(len(data.index)) * 1.2
        bar_height = 0.8
        cumulative = np.zeros(len(data.index))

        for i, level in enumerate(familiarity_levels):
            values = data[level].values
            ax.barh(y_pos, values, left=cumulative, height=bar_height, label=level, color=colors[i])
            cumulative += values

        ax.set_xlabel('', fontweight='bold', fontname='Times New Roman')
        ax.set_title(category, fontweight='bold', fontsize=18, fontname='Times New Roman', pad=20)
        ax.set_xlim(0, 100)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if col == 0:
            ax.set_yticks(y_pos)
            ax.set_yticklabels(data.index, fontweight='bold', fontname='Times New Roman', fontsize=14)
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])

        for i, container in enumerate(ax.containers):
            text_color = 'black' if i < 3 else 'white'
            ax.bar_label(container, label_type='center', fmt='%.1f%%', fontname='Times New Roman',
                         fontweight='bold', fontsize=12, padding=4, color=text_color)

        ax.set_ylim(y_pos.min() - bar_height/2, y_pos.max() + bar_height/2)

    # Remove any unused subplots
    for idx in range(num_categories_adjusted, num_rows * graphs_per_row):
        row = idx // graphs_per_row
        col = idx % graphs_per_row
        if num_rows > 1:
            fig.delaxes(axes[row, col])
        elif graphs_per_row > 1:
            fig.delaxes(axes[col])

    legend_patches = [mpatches.Patch(color=color, label=level) for color, level in zip(colors, familiarity_levels)]
    fig.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.01),
               ncol=min(graphs_per_row, 5), fontsize=12, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15 if num_rows > 1 else 0.25, hspace=0.6 if num_rows > 1 else 0.25, wspace=0.2)

    safe_demographic = re.sub(r'[<>:"/\\|?*]', '_', demographic)
    plt.savefig(f'figures/familiarity_analysis/governance_familiarity/C.2_{safe_demographic}_combined.png', dpi=300, bbox_inches='tight')
    plt.close()

def process_and_create_charts(df, demographic):
    all_data = {}

    if demographics[demographic]['mapping']:
        categories = set(demographics[demographic]['mapping'].values())
        df['GroupedCategory'] = df.iloc[:, demographics[demographic]['column']].map(demographics[demographic]['mapping'])
    else:
        categories = df.iloc[:, demographics[demographic]['column']].dropna().unique()
        df['GroupedCategory'] = df.iloc[:, demographics[demographic]['column']]

    for category in categories:
        df_category = df[df['GroupedCategory'] == category]
        data = pd.DataFrame(index=regulation_names, columns=familiarity_levels)

        for idx, initiative in enumerate(regulation_names):
            if idx < len(regulation_columns):
                column = regulation_columns[idx]
                if column < df_category.shape[1]:
                    valid_responses = df_category.iloc[:, column].dropna()
                    total_respondents = len(valid_responses)
                    if total_respondents > 0:
                        counts = valid_responses.value_counts()
                        for level, count in counts.items():
                            if level in familiarity_levels:
                                data.loc[initiative, level] = (count / total_respondents) * 100
                    else:
                        print(f"Column index {column} is out of range for {initiative} in {category}")

        data = data.fillna(0)
        data = data[familiarity_levels]
        all_data[category] = data
        create_chart(data, category, demographic)

    return all_data

def print_stats(all_data, demographic):
    stats_summary = f"Statistics for {demographic}:\n\n"
    for category, data in all_data.items():
        stats_summary += f"{category}:\n"
        total_respondents = data.iloc[0].sum() / 100
        for initiative in regulation_names:
            stats_summary += f"{initiative}:\n"
            for level in familiarity_levels:
                percentage = data.loc[initiative, level]
                count = round(percentage * total_respondents / 100)
                stats_summary += f" {level}: {count:.0f} ({percentage:.1f}%),"
            stats_summary += "\n"
        stats_summary += f"Total respondents: {total_respondents:.0f}\n" + "="*50 + "\n"

    # Instead of writing to a text file, we simply print the summary
    print(stats_summary)

def process_combined_data(df):
    all_data = {}
    data = pd.DataFrame(index=regulation_names, columns=familiarity_levels)

    for idx, initiative in enumerate(regulation_names):
        if idx < len(regulation_columns):
            column = regulation_columns[idx]
            if column < df.shape[1]:
                valid_responses = df.iloc[:, column].dropna()
                total_respondents = len(valid_responses)
                if total_respondents > 0:
                    counts = valid_responses.value_counts()
                    for level, count in counts.items():
                        if level in familiarity_levels:
                            data.loc[initiative, level] = (count / total_respondents) * 100

    data = data.fillna(0)
    data = data[familiarity_levels]
    all_data['All Participants'] = data
    create_chart(data, 'All Participants', 'Combined')
    return all_data

# Main execution
df = read_and_clean_csv(file_path)
if df is not None:
    print("Processing all participants...")
    combined_data = process_combined_data(df)
    print_stats(combined_data, 'All Participants')

    for demographic in demographics:
        print(f"Processing {demographic}...")
        all_data = process_and_create_charts(df, demographic)
        # Generate combined charts across different categories within a demographic, excluding 'Other' for 'Role'
        categories_to_exclude = ['Other'] if demographic == 'Role' else []
        create_combined_chart(all_data, demographic, graphs_per_row=3, categories_to_exclude=categories_to_exclude)
        print_stats(all_data, demographic)

    print("All graphs have been saved in the designated folder.")


# 
# ### Rank Demographics Familiarity with Principles and Regultions (Requires above code to be run first), input demographic at bottom of cell
# 

# In[ ]:


###################
# Rank demographics familiarities with principles
###################

def read_and_clean_csv(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"Successfully read {file_path}.")
        return df
    except UnicodeDecodeError:
        print(f"Failed to read {file_path} with utf-8 encoding.")
        return None

def rank_demographics_by_principle(df, track, demographic_key):
    ranked_data = {}

    # Get the demographic details from the selected key
    demographic = demographics[demographic_key]
    print(f"\nRanking {demographic_key} demographics for Track {track}...\n")

    # Process each principle and demographic to calculate the percentage for "Extremely Familiar" and "Moderately Familiar"
    for principle_idx, principle in enumerate(principles):
        ranked_data[principle] = []

        demographic_data = {category: 0 for category in set(demographic['mapping'].values())}  # Initialize with 0 for all categories

        df['GroupedCategory'] = df.iloc[:, demographic['column']].map(demographic['mapping'])

        # Iterate over each category within the demographic (e.g., 1-5 Employees, 6-20 Employees, etc.)
        for category in set(demographic['mapping'].values()):
            df_category = df[df['GroupedCategory'] == category]
            valid_columns = principle_columns[track]
            df_track = df_category.dropna(subset=[df_category.columns[valid_columns[principle_idx]]])

            if not df_track.empty:
                # Count "Extremely Familiar" and "Moderately Familiar" responses
                counts = df_track.iloc[:, valid_columns[principle_idx]].value_counts()
                extremely_familiar = counts.get('Extremely Familiar', 0)
                moderately_familiar = counts.get('Moderately Familiar', 0)
                somewhat_familiar = counts.get('Somewhat Familiar', 0)
                total_respondents = len(df_track)

                total_familiar = ((extremely_familiar + moderately_familiar + somewhat_familiar) / total_respondents) * 100
                demographic_data[category] = total_familiar

        # Rank the categories for this demographic and principle
        ranked_categories = sorted(demographic_data.items(), key=lambda x: x[1], reverse=True)
        ranked_data[principle].append((demographic_key, ranked_categories))

    return ranked_data

def display_ranked_data(ranked_data):
    for principle, demographic_rankings in ranked_data.items():
        print(f"\n### {principle} Rankings ###\n")
        for demographic, rankings in demographic_rankings:
            print(f"{demographic}:")
            for category, percentage in rankings:
                if percentage > 0:  # Only display categories with responses
                    print(f"  {category}: {percentage:.1f}%")
            print("\n")

# Main execution
df = read_and_clean_csv(file_path)
if df is not None:
    # Select the demographic to rank (e.g., 'Company Size')
    selected_demographic = 'Location' # This can be changed to other demographic keys like 'Location', etc.

    for track in ['B']:
        print(f"Processing Track {track} for {selected_demographic}...")
        ranked_data = rank_demographics_by_principle(df, track, selected_demographic)
        display_ranked_data(ranked_data)


# In[ ]:


###################
# Rank demographics familiarities with regulations
###################

def rank_by_familiarity_and_average(df, demographic):
    # Ensure the demographic is valid
    if demographic not in demographics:
        print(f"Invalid demographic: {demographic}")
        return
    
    # Prepare to store the ranking data and average scores
    ranking_data = {}
    avg_familiarity_by_category = {}

    # Group by the selected demographic category
    df['GroupedCategory'] = df.iloc[:, demographics[demographic]['column']].map(demographics[demographic]['mapping'])

    for initiative in regulation_names:
        initiative_scores = {}

        # For each category within the demographic (e.g., US, Europe for Location)
        for category in set(demographics[demographic]['mapping'].values()):
            df_category = df[df['GroupedCategory'] == category]
            if df_category.empty:
                continue
            
            # Calculate the percentage of respondents for "Extremely Familiar", "Moderately Familiar", and "Somewhat Familiar"
            total_respondents = len(df_category)
            if total_respondents == 0:
                continue
            
            familiarity_count = df_category.iloc[:, regulation_names.index(initiative) + regulation_columns[0]].value_counts()
            at_least_somewhat_familiar = familiarity_count.get('Extremely Familiar', 0) + familiarity_count.get('Moderately Familiar', 0) + familiarity_count.get('Somewhat Familiar', 0)
            
            percentage = (at_least_somewhat_familiar / total_respondents) * 100
            initiative_scores[category] = percentage

            # Add score to the average familiarity for each category
            if category not in avg_familiarity_by_category:
                avg_familiarity_by_category[category] = []
            avg_familiarity_by_category[category].append(percentage)
        
        # Sort the categories for this regulation based on the percentage
        sorted_scores = sorted(initiative_scores.items(), key=lambda x: x[1], reverse=True)
        ranking_data[initiative] = sorted_scores

    # Print out the ranking results for each regulation
    for initiative, rankings in ranking_data.items():
        print(f"\nRanking for {initiative} based on '{demographic}':")
        for idx, (category, score) in enumerate(rankings, start=1):
            print(f"{idx}. {category}: {score:.2f}% at least somewhat familiar")

    # Calculate and print the average familiarity score for each category across all principles
    avg_familiarity_scores = {
        category: sum(scores) / len(scores) for category, scores in avg_familiarity_by_category.items()
    }

    # Sort the categories based on their average scores
    sorted_avg_scores = sorted(avg_familiarity_scores.items(), key=lambda x: x[1], reverse=True)

    print(f"\nAverage familiarity across all principles for {demographic}:")
    for idx, (category, avg_score) in enumerate(sorted_avg_scores, start=1):
        print(f"{idx}. {category}: {avg_score:.2f}% average familiarity")

# Example usage:
demographic_input = 'Location'  # Change to the desired demographic to analyze
rank_by_familiarity_and_average(df, demographic_input)


# ### Familiarity Heatmaps 
# 
# 

# In[ ]:


####################################################################
# Build heatmaps of the regions to their familiarity of principles and regulations
# Survey Questions (P8) -> (B.2.1 + A.1.1) combined and (P10) -> Regulations
###################################################################
# Load the CSV file into a DataFrame
df = pd.read_csv('accepted_maybe_responses.csv')

# Use the 28th column for the region (index 27 because Python uses 0-based indexing)
region_column = df.columns[28]

# Use columns CB through CJ (79:88) and AU through BC (46:55) for Likert scale responses (principles)
likert_columns_1 = df.columns[79:88]  # CB through CJ
likert_columns_2 = df.columns[46:55]  # AU through BC

# Use columns HM through HT (220:228) for Likert scale responses (regulations)
regulation_columns = df.columns[220:228]  # HM through HT

# Create a mapping for the Likert scale responses to numeric values
likert_mapping = {
    "Not Familiar At All": 1,
    "Slightly Familiar": 2,
    "Somewhat Familiar": 3,
    "Moderately Familiar": 4,
    "Extremely Familiar": 5
}

# Create a mapping for the column names to the principles
principle_mapping = {
    df.columns[79]: "Respect for Human Rights",
    df.columns[80]: "Data Protection and Right to Privacy",
    df.columns[81]: "Harm Prevention and Beneficence",
    df.columns[82]: "Non-Discrimination and Freedom of Privileges",
    df.columns[83]: "Fairness and Justice",
    df.columns[84]: "Transparency and Explainability of AI Systems",
    df.columns[85]: "Accountability and Responsibility",
    df.columns[86]: "Democracy and Rule of Law",
    df.columns[87]: "Environment and Social Responsibility"
}

# Create a mapping for the column names to the regulations
regulation_mapping = {
    df.columns[220]: "European Union Artificial Intelligence Act",
    df.columns[221]: "US Executive Order on Safe, Secure and Trustworthy AI",
    df.columns[222]: "US Algorithmic Accountability Act",
    df.columns[223]: "NIST Technical AI Standards",
    df.columns[224]: "NIST AI Risk Management Framework",
    df.columns[225]: "UN General Assembly's Resolution on AI Systems",
    df.columns[226]: "OECD Principles for Trustworthy AI",
    df.columns[227]: "G20 AI Principles"
}

# Apply the mapping to all Likert scale columns
for column in list(likert_columns_1) + list(likert_columns_2) + list(regulation_columns):
    df[column] = df[column].map(likert_mapping)

# Combine the two sets of Likert columns for principles
for i in range(9):
    combined_column = f'combined_principle_{i}'
    df[combined_column] = df[[likert_columns_1[i], likert_columns_2[i]]].mean(axis=1)

combined_principle_columns = [f'combined_principle_{i}' for i in range(9)]

# Function to create and display heatmap
def plot_heatmap(data, title, xlabel):
    # Remove rows with all NaN values
    data_clean = data.dropna(how='all')
    
    if data_clean.empty:
        print(f"No data to plot for: {title}")
        return
    
    plt.figure(figsize=(20, 12))
    sns.heatmap(data_clean, annot=True, cmap='coolwarm', linewidths=.5, fmt=".2f", cbar_kws={'label': 'Familiarity Level'})
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Region / Country')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Calculate and plot for principles
average_familiarity_by_region_principles = df.groupby(region_column)[combined_principle_columns].mean()
average_familiarity_by_region_principles.columns = [principle_mapping[col] for col in likert_columns_1]
plot_heatmap(average_familiarity_by_region_principles, 'Familiarity with AI Ethics Principles by Region (Combined)', 'AI Ethics Principles')

# Calculate and plot for regulations
average_familiarity_by_region_regulations = df.groupby(region_column)[regulation_columns].mean()
average_familiarity_by_region_regulations = average_familiarity_by_region_regulations.rename(columns=regulation_mapping)
plot_heatmap(average_familiarity_by_region_regulations, 'Familiarity with AI Regulations by Region', 'AI Regulations')

# Create a new DataFrame with grouped regions
def group_regions(region):
    if region in ['North America', 'EU/UK/EEA']:
        return region
    else:
        return 'Other'

grouped_df = df.copy()
grouped_df[region_column] = grouped_df[region_column].apply(group_regions)

# Calculate and plot for grouped principles
grouped_average_familiarity_principles = grouped_df.groupby(region_column)[combined_principle_columns].mean()
grouped_average_familiarity_principles.columns = [principle_mapping[col] for col in likert_columns_1]
plot_heatmap(grouped_average_familiarity_principles, 'Familiarity with AI Ethics Principles by Grouped Regions (Combined)', 'AI Ethics Principles')

# Calculate and plot for grouped regulations
grouped_average_familiarity_regulations = grouped_df.groupby(region_column)[regulation_columns].mean()
grouped_average_familiarity_regulations = grouped_average_familiarity_regulations.rename(columns=regulation_mapping)
plot_heatmap(grouped_average_familiarity_regulations, 'Familiarity with AI Regulations by Grouped Regions', 'AI Regulations')

# Save the data to CSV for further review
average_familiarity_by_region_principles.to_csv('familiarity_by_region_and_principle_combined.csv')
average_familiarity_by_region_regulations.to_csv('familiarity_by_region_and_regulation.csv')
grouped_average_familiarity_principles.to_csv('familiarity_by_grouped_region_and_principle_combined.csv')
grouped_average_familiarity_regulations.to_csv('familiarity_by_grouped_region_and_regulation.csv')


# In[ ]:


####################################################################
# Build heatmaps of the roles to their familiarity with principles and regulations
# Survey Questions (P9) -> (B.2.1 + A.1.1) and (P10) -> Regulations
####################################################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file into a DataFrame
df = pd.read_csv('accepted_maybe_responses.csv')

# Use the 31st column for the role (index 30 because Python uses 0-based indexing)
role_column = df.columns[30]

# Use columns CB through CJ (79:88) and AU through BC (46:55) for Likert scale responses (principles)
likert_columns_1 = df.columns[79:88]  # CB through CJ
likert_columns_2 = df.columns[46:55]  # AU through BC

# Use columns HM through HT (213:221) for Likert scale responses (regulations)
regulation_columns = df.columns[220:228]  # HM through HT

# Create a mapping for the Likert scale responses to numeric values
likert_mapping = {
    "Not Familiar At All": 1,
    "Slightly Familiar": 2,
    "Somewhat Familiar": 3,
    "Moderately Familiar": 4,
    "Extremely Familiar": 5
}

# Create a mapping for the column names to the principles
principle_mapping = {
    0: "Respect for Human Rights",
    1: "Data Protection and Right to Privacy",
    2: "Harm Prevention and Beneficence",
    3: "Non-Discrimination and Freedom of Privileges",
    4: "Fairness and Justice",
    5: "Transparency and Explainability of AI Systems",
    6: "Accountability and Responsibility",
    7: "Democracy and Rule of Law",
    8: "Environment and Social Responsibility"
}

# Create a mapping for the column names to the regulations
regulation_mapping = {
    df.columns[220]: "European Union Artificial Intelligence Act",
    df.columns[221]: "US Executive Order on Safe, Secure and Trustworthy AI",
    df.columns[222]: "US Algorithmic Accountability Act",
    df.columns[223]: "NIST Technical AI Standards",
    df.columns[224]: "NIST AI Risk Management Framework",
    df.columns[225]: "UN General Assembly's Resolution on AI Systems",
    df.columns[226]: "OECD Principles for Trustworthy AI",
    df.columns[227]: "G20 AI Principles"
}

# Apply the mapping to all Likert scale columns
for column in list(likert_columns_1) + list(likert_columns_2) + list(regulation_columns):
    df[column] = df[column].map(likert_mapping)

# Drop rows where the role is empty
df = df.dropna(subset=[role_column])

# Calculate the average of both sets of responses for principles
for i in range(9):
    df[f'combined_principle_{i}'] = df[[likert_columns_1[i], likert_columns_2[i]]].mean(axis=1)

# Calculate the mean familiarity for each role and principle
combined_principle_columns = [f'combined_principle_{i}' for i in range(9)]
average_familiarity_by_role_principles = df.groupby(role_column)[combined_principle_columns].mean()

# Rename the columns to the principles
average_familiarity_by_role_principles.columns = [principle_mapping[i] for i in range(9)]

# Calculate the mean familiarity for each role and regulation
average_familiarity_by_role_regulations = df.groupby(role_column)[regulation_columns].mean()

# Rename the columns to the regulations
average_familiarity_by_role_regulations = average_familiarity_by_role_regulations.rename(columns=regulation_mapping)

# Function to plot heatmap
def plot_heatmap(data, title, ylabel):
    if data.empty:
        print(f"No valid data to plot for {title}. Please check your input data.")
    else:
        # Remove any rows or columns that are all NaN
        data = data.dropna(how='all').dropna(axis=1, how='all')
        
        if data.empty:
            print(f"After removing NaN values, no data remains for {title}. Please check your input data.")
        else:
            plt.figure(figsize=(20, 12))  # Increased figure size
            sns.heatmap(data, annot=True, cmap='coolwarm', linewidths=.5, fmt=".2f", cbar_kws={'label': 'Average Familiarity Level'})
            plt.title(title)
            plt.xlabel('AI Ethics Principles / Regulations')
            plt.ylabel(ylabel)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()

# Plot heatmap for principles
plot_heatmap(average_familiarity_by_role_principles, 'Average Familiarity with AI Ethics Principles by Role (Combined Responses)', 'Role')

# Plot heatmap for regulations
plot_heatmap(average_familiarity_by_role_regulations, 'Average Familiarity with AI Regulations by Role', 'Role')

# Save the pivoted data to CSV for further review
average_familiarity_by_role_principles.to_csv('familiarity_by_role_and_principle_combined.csv')
average_familiarity_by_role_regulations.to_csv('familiarity_by_role_and_regulation.csv')

