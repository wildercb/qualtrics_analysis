# Survey Data and Analysis

This repository contains the survey data, and its analysis for our developer survey to learn about principles of ethical AI. 

- **Excel Files Note:** all excel files that contain survey data we can index by column contain response distributions on the bottom row for each column. 
--- 

## Survey Data (`/data/`)

### Survey Responses
- **Files:**
  - `survey_results.xlsx`
  - `survey_results.pdf`
  - `ai_study_finalized.csv`
- **Description:**  
  These files contain the full accepted responses from the survey. 

####  Quantitative Analysis By Question
- **Files:**
  - `quantitative_analysis.docx`  
    _Figures and textual breakdown of response distribution for multiple choice (quantiative) questions._
  - `quantitative_results.xlsx`  
    _Full response distribution for all quantitative questions._


#### Figures (`/data/figures/`)
- **Description:**  
  Contains figures generated during analysis, including those for raw data analysis and familiarity analysis.  
  Graphs produced from the ethical AI familiarity analysis are saved in `data/figures/familiarity_analysis`.

### Annotations

#### Annotation Files (`/data/annotations/`)
- **Formats:**
  - **JSON:** Annotator JSON Books (`/data/annotations/json/`)
  - **Excel:** Annotations in Excel (`/data/annotations/xcel/`)
- **Description:**  
  Annotations provide responses with themes and codes, aside from those questions about risk and mitigation methods, 
  - **Themes** refer to commonalities in the text which may be mentioned in the paper. 
  - **Codes**  are used to highlight interesting responses.   

---
## Notebooks (`/notebooks/`)

#### Correlation Tests
- **File:** `/notebooks/correlation_tests.ipynb`
- **Description:**  
  Contains correlation tests on the survey data.

#### Familiarity Analysis Notebook
- **File:** `/notebooks/familiarity_analysis.ipynb`
- **Description:**  
  Analyzes data related to familiarity with the principles of ethical AI and current regulatory initiatives.  
  Relevant graphs are saved to `data/figures/familiarity_analysis`.

#### Quantitative Analysis Notebook
- **File:** `/notebooks/quantitative.ipynb`
- **Description:**  
  Builds graphs and generates a DOCX file summarizing all quantitative questions, and the demographics table.

#### Dataset Exploration
- **File:** `data_view.ipynb`
- **Description:**  
  F(x)'s to search dataset, i.e., ranking responses by demographics.

---