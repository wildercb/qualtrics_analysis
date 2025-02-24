from globals import *

import csv
import re
import json
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import plotly.graph_objects as go
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def protect_special_phrases(text: str, special_phrases) -> str:
    """
    Replace the commas in special phrases with a unique token <COMMA>.
    """
    if not text:
        return text
    protected_text = text
    for phrase in special_phrases:
        safe_phrase = phrase.replace(",", "<COMMA>")
        protected_text = protected_text.replace(phrase, safe_phrase)
    return protected_text

def restore_special_phrases(text: str) -> str:
    """
    Restore <COMMA> back to ',' in the text.
    """
    return text.replace("<COMMA>", ",")

def get_demographic_value(row, demo_key, df_columns):
    """
    Return the aggregated demographic category for the row,
    given the 'demographic_key' and the 'demographics' dictionary.
    """
    demo_info = demographics[demo_key]
    col_index = demo_info['column']
    mapping = demo_info['mapping']

    raw_value = row[df_columns[col_index]]  # if your DataFrame is wide-format

    if pd.isna(raw_value) or not str(raw_value).strip():
        return "Other"

    if mapping is not None:
        return mapping.get(raw_value, "Other")
    else:
        # If mapping is None, use the raw value directly
        return str(raw_value)

def rank_question_answers_by_demographic(
    df, 
    question_col, 
    demographic_key,
    special_phrases=None,
    skip_first_row_question_text=False
):
    """
    Given a DataFrame `df` with one row per respondent,
    a `question_col` name (or index) for the question,
    and a `demographic_key` from the 'demographics' dict,
    return a dict of:
       {
         "grouped_by_demo": { 
             <demographic_group>: [(answer, count, pct), ...] sorted by pct desc
         },
         "grouped_by_answer": {
             <answer>: [(demographic_group, count, pct), ...] sorted by pct desc
         }
       }
    so you can see both:
      1) For each demographic, which answers were most popular?
      2) For each answer, which demographics selected it most?
    """

    if special_phrases is None:
        special_phrases = []

    data = df.copy()
    # 1) Filter out rows that have no data for this question
    data = data.dropna(subset=[question_col])

    # 2) If the first row is question text, skip it
    if skip_first_row_question_text and len(data) > 0:
        data = data.iloc[1:]

    # 3) Parse each respondent's answers
    all_answers = []
    for idx, row_i in data.iterrows():
        raw_answer = str(row_i[question_col])
        protected_answer = protect_special_phrases(raw_answer, special_phrases)
        splitted = [x.strip() for x in protected_answer.split(",") if x.strip()]
        restored = [restore_special_phrases(s) for s in splitted]
        all_answers.append(restored)

    data["parsed_answers"] = all_answers

    # 4) Build the aggregated demographic category for each row
    df_columns = list(data.columns)
    data["demo_category"] = data.apply(lambda r: get_demographic_value(r, demographic_key, df_columns), axis=1)

    # 5) group_counts[demo_group][answer] = number_of_mentions
    #    group_respondents[demo_group] = total respondents in that group
    group_counts = defaultdict(lambda: defaultdict(int))
    group_respondents = defaultdict(int)

    for idx, row_i in data.iterrows():
        group = row_i["demo_category"]
        group_respondents[group] += 1
        for ans in row_i["parsed_answers"]:
            group_counts[group][ans] += 1

    # 6) Build the results: grouped_by_demo
    grouped_by_demo = {}
    for group, answers_dict in group_counts.items():
        group_size = group_respondents[group]
        if group_size == 0:
            grouped_by_demo[group] = []
            continue

        # list of (answer, count, pct)
        group_answer_list = []
        for ans, c in answers_dict.items():
            pct = (c / group_size) * 100
            group_answer_list.append((ans, c, pct))

        group_answer_list.sort(key=lambda x: x[2], reverse=True)
        grouped_by_demo[group] = group_answer_list

    # 7) Build the "reverse" view: for each answer, how do the demographics break down?
    #    We'll gather all unique answers from group_counts
    all_answers_set = set()
    for group, answers_dict in group_counts.items():
        for ans in answers_dict.keys():
            all_answers_set.add(ans)

    # answer_breakdown[ans][group] = (count, pct)
    answer_breakdown = defaultdict(lambda: {})
    for ans in all_answers_set:
        for group, answers_dict in group_counts.items():
            group_size = group_respondents[group]
            if group_size == 0:
                # no respondents in this group
                answer_breakdown[ans][group] = (0, 0.0)
            else:
                c = answers_dict.get(ans, 0)
                pct = (c / group_size) * 100
                answer_breakdown[ans][group] = (c, pct)

    # Now convert that to a list-of-tuples format sorted by pct desc
    grouped_by_answer = {}
    for ans, group_dict in answer_breakdown.items():
        group_list = []
        for group, (c, pct) in group_dict.items():
            group_list.append((group, c, pct))
        group_list.sort(key=lambda x: x[2], reverse=True)
        grouped_by_answer[ans] = group_list

    return {
        "grouped_by_demo": grouped_by_demo,
        "grouped_by_answer": grouped_by_answer
    }


#################################################
# Utils for reading our annotations data 
##################################################

def save_fig_as_png_selenium(fig, filename, width=1200, height=700):
    """
    Save a Plotly figure as a PNG:
      1. Write an HTML file,
      2. Open it in a headless Chrome browser,
      3. Take a screenshot.
      
    Parameters:
      fig (go.Figure): The Plotly figure.
      filename (str): PNG filename to save.
      width (int): Window width.
      height (int): Window height.
    """
    html_path = filename.replace(".png", ".html")
    fig.write_html(html_path, include_plotlyjs="cdn", auto_open=False)
    
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    
    driver = webdriver.Chrome(options=chrome_options)
    driver.set_window_size(width, height)
    driver.get("file://" + os.path.abspath(html_path))
    
    driver.save_screenshot(filename)
    driver.quit()
    os.remove(html_path)

def sanitize_filename(name: str) -> str:
    """Replace invalid filename characters with underscores."""
    return re.sub(r'[\\/*?:"<>|]', '_', name)

def generate_annotator_report(input_json, output_graph=True):
    """
    Reads a JSON file and for each question:
      - Prints a textual breakdown including:
          • Total responses.
          • Count (and percentage) of responses with no theme (labeled as "Other").
          • Breakdown of themes (with counts and percentages).
          • Breakdown of codes (with counts and percentages).
      - If output_graph is True, generates a bar chart that shows only the theme distribution
        (i.e. "Other" and each individual theme) and saves it as a PNG in:
            figures/annotations/{input_json_basename}/{question_key}.png
    """
    # Load the JSON data.
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Prepare the output directory for graphs if needed.
    if output_graph:
        base_name = os.path.splitext(os.path.basename(input_json))[0]
        output_dir = os.path.join("figures", "annotations", sanitize_filename(base_name))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    # Process each question.
    for question_key, responses in data.items():
        total = len(responses)
        if total == 0:
            continue
        
        theme_counter = {}
        code_counter = {}
        no_theme_count = 0  # Count responses with an empty theme list
        
        # Count themes and codes.
        for response in responses:
            themes = set(response.get("theme", []))
            codes = set(response.get("codes", []))
            
            if not themes:
                no_theme_count += 1
            
            for theme in themes:
                theme_str = str(theme)
                theme_counter[theme_str] = theme_counter.get(theme_str, 0) + 1
            for code in codes:
                code_str = str(code)
                code_counter[code_str] = code_counter.get(code_str, 0) + 1
        
        # Print textual breakdown.
        print(f"Question: {question_key}")
        print(f"Total responses: {total}")
        print(f"Responses with no theme (Other): {no_theme_count} ({(no_theme_count / total) * 100:.2f}%)")
        
        if theme_counter:
            print("Themes:")
            for theme, count in sorted(theme_counter.items(), key=lambda item: item[1], reverse=True):
                print(f"  - Theme: {theme}: {count} responses ({(count / total) * 100:.2f}%)")
        else:
            print("No themes found.")
        
        if code_counter:
            print("Codes:")
            for code, count in sorted(code_counter.items(), key=lambda item: item[1], reverse=True):
                print(f"  - Code: {code}: {count} responses ({(count / total) * 100:.2f}%)")
        else:
            print("No codes found.")
        print("-" * 40)
        
        # Generate and save the graph only if output_graph is True.
        if output_graph:
            options = []
            counts = []
            percentages = []
            
            # Add "Other" category if applicable.
            if no_theme_count > 0:
                options.append("Other")
                counts.append(no_theme_count)
                percentages.append(round((no_theme_count / total) * 100, 2))
            
            # Add themes, sorted descending by count.
            for theme, count in sorted(theme_counter.items(), key=lambda item: item[1], reverse=True):
                options.append(f"{theme}")
                counts.append(count)
                percentages.append(round((count / total) * 100, 2))
            
            # Prepare text labels for the bars.
            text_vals = [f"{pct}%" for pct in percentages]
            marker_dict = dict(
                color=counts,
                colorscale="Blues",
                showscale=True,
                colorbar=dict(title="Count")
            )
            
            # Create the bar chart with Plotly.
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=options,
                    y=counts,
                    text=text_vals,
                    textposition="outside",
                    textfont=dict(
                        size=18,
                        color="black",
                        family="Arial Black"
                    ),
                    marker=marker_dict,
                    cliponaxis=False
                )
            )
            
            # Compute y-axis range with a 15% buffer.
            max_y = max(counts) if counts else 0
            y_buffer = max_y * 0.15
            
            fig.update_layout(
                xaxis_title="Response Option",
                yaxis_title="Number of Responses",
                template="plotly_white",
                xaxis=dict(
                    tickangle=45,
                    tickfont=dict(
                        family="Arial Black",
                        size=12,
                        color="black"
                    )
                ),
                margin=dict(l=80, r=80, t=80, b=80),
            )
            fig.update_yaxes(range=[0, max_y + y_buffer])
            
            # Create a safe filename for the question.
            question_filename = sanitize_filename(question_key)
            image_path = os.path.join(output_dir, f"{question_filename}.png")
            
            # Save the figure as a PNG (via Selenium screenshot).
            save_fig_as_png_selenium(fig, image_path, width=1200, height=700)
            print(f"Saved graph for question {question_key} to {image_path}\n")

def search_annotations_texts_by_tag(input_json, tag):
    """
    Reads the JSON file and returns all texts (with their question code) where the specified tag appears.
    The tag is searched for in both the "theme" and "codes" lists of each response.
    
    Parameters:
      input_json (str): Path to the input JSON file.
      tag (str): The theme or code to search for.
    
    Returns:
      List of tuples in the form (question_code, text) for each matching response.
    """
    matching_texts = []
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for question_key, responses in data.items():
        # Extract question code (everything before the first colon)
        q_code = question_key.split(":", 1)[0].strip()
        for response in responses:
            themes = set(response.get("theme", []))
            codes = set(response.get("codes", []))
            if tag in themes or tag in codes:
                text = response.get("text", "")
                matching_texts.append((q_code, text))
    
    return matching_texts



def analyze_demographic_themes(json_path, demographic_key, csv_path):
    """
    Loads a JSON file (where the order of responses in each question corresponds to the CSV rows)
    and a CSV file, then for the given demographic (e.g., "Role") produces three outputs:
    
      1. Aggregated Distribution of Themes by Demographic:
         For each role, the overall count and percentage (across all questions) for each theme.
         
      2. Per-Theme Question Ranking by Demographic:
         For each role and each theme, select the single question (across all questions) where that theme’s
         percentage (within that role for that question) is highest.
         
      3. Ranking of Roles by Theme:
         For each theme (aggregated across roles), rank the roles by their aggregated percentage for that theme,
         and include the count.
    
    Note: The CSV file’s first row is used for column headers, but for data rows we skip the first two rows.
    
    Parameters:
        json_path (str): Path to the JSON file containing questions and responses.
                         (Order of responses must match CSV rows.)
        demographic_key (str): The demographic key (e.g., "Role").
        csv_path (str): Path to the CSV file containing demographic data.
    
    Returns:
        dict: {
            "aggregated_distribution": { role: { "total_responses": N, "themes": { theme: {"count": n, "percentage": p}, ... } } },
            "per_theme_question_ranking": { role: { theme: { "question": best_question, "percentage": best_pct } } },
            "per_theme_role_ranking": { theme: [ { "role": role, "percentage": p, "count": n }, ... sorted descending ] }
        }
    """
    if demographic_key not in demographics:
        print(f"Demographic '{demographic_key}' not found in the demographics mapping.")
        return None

    demo_info = demographics[demographic_key]
    column_index = demo_info['column']
    mapping = demo_info.get('mapping')
    
    # Load JSON data.
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Load CSV data.
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        csv_all = list(reader)
    if not csv_all:
        print("CSV file is empty.")
        return None
    
    # Use the first row as header.
    csv_headers = csv_all[0]
    # Skip the first two rows for all other purposes.
    csv_data_rows = csv_all[2:]
    
    # Get header value for the demographic column (to skip accidental header rows)
    header_value = csv_headers[column_index].strip() if column_index < len(csv_headers) else ""
    
    aggregated = {}   # { role: { "total_responses": int, "theme_counts": { theme: count } } }
    per_question = {} # { question: { role: { "total_responses": int, "theme_counts": { theme: count } } } }
    
    for question, responses in json_data.items():
        if question not in per_question:
            per_question[question] = {}
        for i, response in enumerate(responses):
            if i >= len(csv_data_rows):
                continue
            csv_row = csv_data_rows[i]
            if column_index >= len(csv_row):
                continue
            raw_value = csv_row[column_index].strip()
            role_value = mapping.get(raw_value, raw_value) if mapping is not None else raw_value
            if role_value == header_value:
                continue
            
            if role_value not in aggregated:
                aggregated[role_value] = {"total_responses": 0, "theme_counts": {}}
            aggregated[role_value]["total_responses"] += 1
            themes = response.get("theme", [])
            if not isinstance(themes, list):
                themes = [themes]
            for theme in themes:
                theme = theme.strip()
                if theme:
                    aggregated[role_value]["theme_counts"][theme] = aggregated[role_value]["theme_counts"].get(theme, 0) + 1
            
            if role_value not in per_question[question]:
                per_question[question][role_value] = {"total_responses": 0, "theme_counts": {}}
            per_question[question][role_value]["total_responses"] += 1
            for theme in themes:
                theme = theme.strip()
                if theme:
                    per_question[question][role_value]["theme_counts"][theme] = per_question[question][role_value]["theme_counts"].get(theme, 0) + 1

    aggregated_distribution = {}
    for role_value, data in aggregated.items():
        total = data["total_responses"]
        theme_dist = {}
        for theme, count in data["theme_counts"].items():
            pct = (count / total * 100) if total > 0 else 0
            theme_dist[theme] = {"count": count, "percentage": round(pct, 2)}
        aggregated_distribution[role_value] = {
            "total_responses": total,
            "themes": dict(sorted(theme_dist.items(), key=lambda x: x[1]['count'], reverse=True))
        }

    per_theme_question_ranking = {}
    for question, group_data in per_question.items():
        for role_value, stats in group_data.items():
            total = stats["total_responses"]
            if total == 0:
                continue
            for theme, count in stats["theme_counts"].items():
                pct = (count / total * 100)
                if role_value not in per_theme_question_ranking:
                    per_theme_question_ranking[role_value] = {}
                if theme not in per_theme_question_ranking[role_value] or pct > per_theme_question_ranking[role_value][theme]["percentage"]:
                    per_theme_question_ranking[role_value][theme] = {"question": question, "percentage": round(pct, 2)}

    per_theme_role_ranking = {}
    for role_value, agg_data in aggregated_distribution.items():
        for theme, theme_data in agg_data["themes"].items():
            if theme not in per_theme_role_ranking:
                per_theme_role_ranking[theme] = []
            per_theme_role_ranking[theme].append({
                "role": role_value,
                "percentage": theme_data["percentage"],
                "count": theme_data["count"]
            })
    for theme in per_theme_role_ranking:
        per_theme_role_ranking[theme].sort(key=lambda x: x["percentage"], reverse=True)

    # ----- PRINTING THE RESULTS -----
    print("=== Aggregated Distribution of Themes by Demographic ===")
    for role_value, info in aggregated_distribution.items():
        print(f"{demographic_key} - {role_value}: Total Responses = {info['total_responses']}")
        for theme, stats in info["themes"].items():
            print(f"  Theme: {theme} at {stats['percentage']}% (Count: {stats['count']})")
        print("")
    '''
    print("=== Per-Theme Question Ranking by Demographic ===")
    for role_value, theme_dict in per_theme_question_ranking.items():
        print(f"\n{demographic_key} - {role_value}:")
        for theme, entry in theme_dict.items():
            print(f"  Theme: {theme} --> Question: {entry['question']} at {entry['percentage']}%")
    '''
    print("\n=== Ranking of Roles by Theme ===")
    for theme, role_list in per_theme_role_ranking.items():
        print(f"\n{theme}:")
        for entry in role_list:
            print(f"  Role: {entry['role']} at {entry['percentage']}% (Count: {entry['count']})")
    
    return {
        "aggregated_distribution": aggregated_distribution,
        "per_theme_question_ranking": per_theme_question_ranking,
        "per_theme_role_ranking": per_theme_role_ranking
    }



def create_principle_role_graph(json_file_path, role_names, principle_names):
    """
    Reads a JSON file, processes data, and creates a table-like graph
    representing principles by roles. Each cell contains horizontal bar graphs
    for themes (above) and codes (below) (separated by a red line in the center),
    showing percentages, with separate color-coded legends for themes and individual codes.
    Allows for custom role and principle names using dictionaries.
    Combines themes and codes regardless of capitalization, without data cleaning.
    """
    try:
        with open(json_file_path, 'r', encoding='latin-1') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {json_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_file_path}")
        return

    extracted_data = []
    for key, values in data.items():
        parts = key.split(":")
        if len(parts) > 1:
            role_principle = parts[0].split(".")
            if len(role_principle) == 3 and role_principle[0] == 'B':
                role = role_principle[1]
                principle = role_principle[2]

                for item in values:
                    themes = item.get('theme', [])
                    codes = [c for c in item.get('codes', []) if c != "HIGHLIGHTER"]

                    # Handle cases where themes or codes are missing or empty
                    if not themes:
                        themes = ['N/A']
                    if not codes:
                        codes = ['N/A']

                    # Combine themes and codes regardless of capitalization
                    for theme in themes:
                        for code in codes:
                            extracted_data.append({
                                'role': role,
                                'principle': principle,
                                'theme': theme.lower(),  # Convert to lowercase
                                'codes': code.lower()   # Convert to lowercase
                            })

    df = pd.DataFrame(extracted_data)

    if df.empty:
        print("No valid data found for plotting.")
        return

    roles = sorted(df['role'].unique())
    principles = sorted(df['principle'].unique(), key=lambda x: int(x))
    num_roles = len(roles)
    num_principles = len(principles)

    # --- Color Mapping ---
    all_themes = df['theme'].unique()
    all_codes = df['codes'].unique()
    num_themes = len(all_themes)
    num_codes = len(all_codes)

    theme_colors = plt.cm.tab20(np.linspace(0, 1, num_themes))
    code_colors = plt.cm.tab20b(np.linspace(0, 1, num_codes))

    theme_color_map = {theme: color for theme, color in zip(all_themes, theme_colors)}
    code_color_map = {code: color for code, color in zip(all_codes, code_colors)}

    # --- Plotting ---
    fig, axes = plt.subplots(num_roles, num_principles, figsize=(25, 15), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.1, hspace=0.4)

    for i, role in enumerate(roles):
        for j, principle in enumerate(principles):
            ax = axes[i, j] if num_roles > 1 else axes[j]

            cell_data = df[(df['role'] == role) & (df['principle'] == principle)]

            # --- Calculate Percentages ---
            theme_counts = cell_data.groupby('theme').size()
            total_themes = theme_counts.sum()
            theme_percentages = (theme_counts / total_themes) * 100 if total_themes > 0 else theme_counts

            code_counts = cell_data.groupby('codes').size()
            total_codes = code_counts.sum()
            code_percentages = (code_counts / total_codes) * 100 if total_codes > 0 else code_counts

            # --- Determine the order of plotting ---
            num_themes = len(theme_percentages)
            num_codes = len(code_percentages)
            y_positions = np.arange(num_themes + num_codes)

            # --- Plot Themes (Horizontal Bars) ---
            y_theme = y_positions[:num_themes]
            theme_bars = ax.barh(y_theme, theme_percentages.values, height=0.4,
                                color=[theme_color_map[theme] for theme in theme_percentages.index],
                                edgecolor='black', linewidth=0.5, label='Themes')

            # --- Plot Codes (Horizontal Bars) ---
            y_code = y_positions[num_themes:]
            code_bars = ax.barh(y_code, code_percentages.values, height=0.4,
                                color=[code_color_map[code] for code in code_percentages.index],
                                edgecolor='black', linewidth=0.5, label='Codes')

            # --- Add Red Separator Line in the Middle ---
            midpoint = num_themes
            ax.axhline(midpoint, color='red', linewidth=1.5)

            # --- Cell-Specific Customization ---
            ax.set_xlim(0, 100)
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.tick_params(axis='x', labelsize=8)
            ax.set_yticks([])

            for spine in ['top', 'bottom', 'left', 'right']:
                ax.spines[spine].set_visible(True)
                ax.spines[spine].set_linewidth(0.5)
                ax.spines[spine].set_color('black')

            # --- Use Provided Names for Roles and Principles ---
            if i == 0:
                principle_label = principle_names.get(principle, f"Principle {principle}")
                ax.set_title(principle_label, fontsize=10, fontweight='bold')
            if j == 0:
                role_label = role_names.get(role, f"Role {role}")
                ax.set_ylabel(role_label, fontsize=10, rotation=0, ha='right', va='center', fontweight='bold')

    # --- Global Customization ---
    fig.suptitle('AI Ethics Principles by Roles: Themes and Codes Distribution', fontsize=16, fontweight='bold')

    # --- Separate Legends for Themes and Codes ---
    theme_handles = [plt.Rectangle((0, 0), 1, 1, color=theme_color_map[theme], edgecolor='black', linewidth=0.5)
                     for theme in all_themes]
    theme_labels = [f"{theme}" for theme in all_themes]

    code_handles = [plt.Rectangle((0, 0), 1, 1, color=code_color_map[code], edgecolor='black', linewidth=0.5)
                    for code in all_codes]
    code_labels = [f"{code}" for code in all_codes]

    # --- Place Legends ---
    legend_y = 0.08
    legend_ncol = 3

    # Theme Legend
    fig.legend(theme_handles, theme_labels, loc='upper center', bbox_to_anchor=(0.3, legend_y),
               fontsize=8, title="Themes", ncol=legend_ncol)

    # Code Legend
    fig.legend(code_handles, code_labels, loc='upper center', bbox_to_anchor=(0.7, legend_y),
               fontsize=8, title="Codes", ncol=legend_ncol)

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])

    plt.show()


def print_analysis_by_principle(df, principle_names):
    """
    Prints a text summary of the theme and code distribution for each principle.
    """
    for principle in sorted(df['principle'].unique()):
        principle_label = principle_names.get(principle, f"Principle {principle}")
        print(f"----- {principle_label} -----")

        principle_data = df[df['principle'] == principle]

        theme_counts = principle_data.groupby('theme').size()
        total_themes = theme_counts.sum()
        theme_percentages = (theme_counts / total_themes) * 100 if total_themes > 0 else theme_counts

        code_counts = principle_data.groupby('codes').size()
        total_codes = code_counts.sum()
        code_percentages = (code_counts / total_codes) * 100 if total_codes > 0 else code_counts

        print("Themes:")
        for theme, percentage in theme_percentages.items():
            print(f"  - {theme}: {percentage:.2f}%")

        print("\nCodes:")
        for code, percentage in code_percentages.items():
            print(f"  - {code}: {percentage:.2f}%")
        print("\n")

def print_analysis_by_role(df, role_names):
    """
    Prints a text summary of the theme and code distribution for each role.
    """
    for role in sorted(df['role'].unique()):
        role_label = role_names.get(role, f"Role {role}")
        print(f"----- {role_label} -----")

        role_data = df[df['role'] == role]

        theme_counts = role_data.groupby('theme').size()
        total_themes = theme_counts.sum()
        theme_percentages = (theme_counts / total_themes) * 100 if total_themes > 0 else theme_counts

        code_counts = role_data.groupby('codes').size()
        total_codes = code_counts.sum()
        code_percentages = (code_counts / total_codes) * 100 if total_codes > 0 else code_counts

        print("Themes:")
        for theme, percentage in theme_percentages.items():
            print(f"  - {theme}: {percentage:.2f}%")

        print("\nCodes:")
        for code, percentage in code_percentages.items():
            print(f"  - {code}: {percentage:.2f}%")
        print("\n")
        
def print_analysis_by_role_and_principle(df, role_names, principle_names):
    """
    Prints a text summary of the theme and code distribution for each role and principle combination.
    """
    for role in sorted(df['role'].unique()):
        for principle in sorted(df['principle'].unique()):
            cell_data = df[(df['role'] == role) & (df['principle'] == principle)]
            role_label = role_names.get(role, f"Role {role}")
            principle_label = principle_names.get(principle, f"Principle {principle}")

            print(f"----- {role_label}, {principle_label} -----")

            theme_counts = cell_data.groupby('theme').size().reset_index(name='count')
            total_themes = theme_counts['count'].sum()
            theme_counts['percentage'] = (theme_counts['count'] / total_themes) * 100 if total_themes > 0 else 0

            code_counts = cell_data.groupby('codes').size().reset_index(name='count')
            total_codes = code_counts['count'].sum()
            code_counts['percentage'] = (code_counts['count'] / total_codes) * 100 if total_codes > 0 else 0

            print("Themes:")
            if not theme_counts.empty:
                for index, row in theme_counts.iterrows():
                    print(f"  - {row['theme']}: {row['percentage']:.2f}%")
            else:
                print("  - N/A")

            print("\nCodes:")
            if not code_counts.empty:
                for index, row in code_counts.iterrows():
                    print(f"  - {row['codes']}: {row['percentage']:.2f}%")
            else:
                print("  - N/A")

            print("\n")

def print_role_principle_analysis(input_json):
    with open(input_json, 'r', encoding='latin-1') as f:
        data = json.load(f)
    extracted_data = []
    for key, values in data.items():
        parts = key.split(":")
        if len(parts) > 1:
            role_principle = parts[0].split(".")
            if len(role_principle) == 3 and role_principle[0] == 'B':
                role = role_principle[1]
                principle = role_principle[2]

                for item in values:
                    themes = item.get('theme', [])
                    codes = [c for c in item.get('codes', []) if c != "HIGHLIGHTER"]

                    if not themes:
                        themes = ['N/A']
                    if not codes:
                        codes = ['N/A']
                    for theme in themes:
                        for code in codes:
                            extracted_data.append({
                                'role': role,
                                'principle': principle,
                                'theme': theme.lower(),
                                'codes': code.lower()
                            })

    df = pd.DataFrame(extracted_data)

    print("Analysis by Principle:")
    print_analysis_by_principle(df, principle_names)

    print("Analysis by Role:")
    print_analysis_by_role(df, role_names)

    print("Analysis by Role and Principle:")
    print_analysis_by_role_and_principle(df, role_names, principle_names)