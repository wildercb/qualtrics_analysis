import os


project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
data_dir = os.path.join(project_root, "data")
survey_data = os.path.join(data_dir, "survey_finalized.csv")

familiarity_levels = [
    'Extremely Familiar',
    'Moderately Familiar',
    'Somewhat Familiar',
    'Slightly Familiar',
    'Not Familiar At All'
]

familiarity_mapping = {
    'Extremely Familiar': 5,
    'Moderately Familiar': 4,
    'Somewhat Familiar': 3,
    'Slightly Familiar': 2,
    'Not Familiar At All': 1
}

importance_mapping = {
    'Not at all important': 1,
    'Not Very Important': 2,
    'Somewhat Important': 3,
    'Moderately Important': 4,
    'Extremely Important': 5
}

confidence_mapping = {
    'Not At All Confident': 1,
    'Not Very Confident': 2,
    'Somewhat Confident': 3,
    'Moderately Confident': 4,
    'Extremely Confident': 5
}

principles = [
    'Respect for Human Rights',
    'Data Protection and Right to Privacy',
    'Harm Prevention and Beneficence',
    'Non-Discrimination and Freedom of Privileges',
    'Fairness and Justice',
    'Transparency and Explainability of AI Systems',
    'Accountability and Responsibility',
    'Democracy and Rule of Law',
    'Environment and Social Responsibility'
]

principle_names = {
    '1': 'Respect for Human Rights',
    '2': 'Data Protection and Privacy',
    '3': 'Harm Prevention and Beneficence',
    '4': 'Non-Discrimination and Freedom',
    '5': 'Fairness and Justice',
    '6': 'Transparency and Explainability',
    '7': 'Accountability and Responsibility',
    '8': 'Democracy and Rule of Law',
    '9': 'Environment and Social Responsibility'
}

# Regulation names corresponding to the columns (truncate or pad if necessary)
regulation_names = [
    'European Union Artificial Intelligence Act',
    'US Executive Order on "Safe, Secure and Trustworthy Artificial Intelligence"',
    'US Algorithmic Accountability Act',
    'NIST Technical AI Standards',
    'NIST AI Risk Management Framework',
    'UN General Assembly Resolution on AI for Sustainable Development',
    'OECD Principles for Trustworthy AI',
    'G20 AI Principles'
]


# Columns related to familiarity with AI ethics principles (A.2.1 and B.2.1)
principle_columns = {
    'A': list(range(46, 55)),  # Columns 46-54 for Track A
    'B': list(range(79, 88))   # Columns 79-87 for Track B
}
regulation_columns = list(range(220, 229))

role_names = {
    '3': 'Account Manager',
    '4': 'Risk Analyst',
    '5': 'Application Developer',
    '6': 'Quality Assurance',
    '7': 'InfoSec',
    '8': 'Application Engineer',
    '15': 'Academic Researcher'
}

# Demographics and the column number they are found in ai_study_finalized
demographics = {
    'Location': {
        'column': 17,
        'mapping': {
            'North America': 'North America',
            'Central/South America': 'Other',
            'EU/UK/EEA': 'Europe',
            'Europe - Outside of EU/UK/EEA': 'Europe',
            'Africa': 'Other',
            'Middle East': 'Other',
            'Asia': 'Other',
            'Australia and Oceania': 'Other',
            'Prefer not to say': 'Other',
            'Other, please specify': 'Other'
        }
    },
    'Age': {
        'column': 142,
        'mapping': {
            '18-24': '18-24',
            '25-34': '25-34'',
            '35-44': '35-44',
            '45-54': '45-54',
            '55-64': '55-64',
            '65 and above': '65+',
            'I prefer not to say': 'Prefer Not to Say',
        }
    },
    'Gender': {
    'column': 'Q13.2',
    'mapping': {
        'Male': 'Male',
        'Female': 'Female',
        'Non-Binary': 'Non-Binary',
        'Other': 'Other',
        'Prefer not to answer': 'Prefer not to answer'
        }
    },
    'Education': {
            'column': 'Q13.3',
            'mapping': {
                'High School': 'High School',
                'Associate Degree': 'Associate Degree',
                'Bachelors Degree': 'Bachelors Degree',
                'Graduate Degree': 'Graduate Degree',
                'Professional Degree': 'Professional Degree',
                'I prefer not to say': 'I prefer not to say',
                'Others (Please Specify)': 'Others (Please Specify)'
            }
        },
        'Education_Field': {
    'column': 'Q13.4',
    'mapping': {
        'Business Administration': 'Business Administration',
        'Computer Science': 'Computer Science',
        'Computer and Electrical Engineering': 'Computer and Electrical Engineering',
        'Data Science': 'Data Science',
        'Information Science': 'Information Science',
        'Information Security': 'Information Security',
        'Information Technology': 'Information Technology',
        'Law': 'Law',
        'Other (please specify)': 'Other (please specify)'
        }
    },
    'Religion': {
    'column': 'Q13.5',
    'mapping': {
        'Christianity': 'Christianity',
        'Islam': 'Islam',
        'Hinduism': 'Hinduism',
        'Buddhism': 'Buddhism',
        'Judaism': 'Judaism',
        'Others (Please Specify)': 'Others (Please Specify)',
        'None': 'None',
        'Prefer not to say': 'Prefer not to say'
        }
    },
    'Religious_Importance': {
    'column': 'Q13.6',
    'mapping': {
        'Not Important': 'Not Important',
        'Slightly Important': 'Slightly Important',
        'Very Important': 'Very Important',
        'Prefer not to say': 'Prefer not to say',
        'Others (Please Specify)': 'Others (Please Specify)'
        }
    }
}

SPECIAL_PHRASES = [
    "other, please specify", "Master's Degree (i.e., MSc., M.A., etc.)", "Information Science (e.g., IT or MIS)", "Administrative role (CEO, Chief Technical Officer, Chief Operating Officer, Chief Information Officer)", "Scrum Master, Product Manager, or Project Manager", "(Software) Developer, Designer, or Architect", "Chatbots, Personal Assistants or Recommender Systems", "Programming Analysis (e.g., Code Completion or Code Generation)", "Model provider APIs, i.e., OpenAI, Anthropic, DeepInfra, LLama", "RNN, LSTM / GRU", "Clean Data to Remove, Mitigate, or Minimize Biases", "Integrating Privacy Enhancing Technologies (PETs) in AI systems (e.g., differential privacy algorithms, federated learning models, etc.)", "Continuously, as part of an ongoing process", "Professional organizations (e.g., ACM, IEEE, ACL, AAAI, etc.)",
]
