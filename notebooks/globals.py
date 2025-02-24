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
    'Company Type': {
        'column': 26,
        'mapping': {
            'Multi-national Corporate': 'Multi-national',
            'Startup/Small Business': 'Startup/Small',
            'Academic Institution/Research Center': 'Academic/Research',
            'Government': 'Government',
            'Individual': 'Individual',
        }
    },
    'Role': {
        'column': 30,
        'mapping': {
            'Administrative role (CEO, Chief Technical Officer, Chief Operating Officer, Chief Information Officer)': 'AI Manager',
            'AI Manager': 'AI Manager',
            'Requirements Analyst or Engineer': 'Requirements analyst',
            'Scrum Master, Product Manager, or Project Manager': 'Requirements analyst',
            'AI Engineer or Developer': 'AI developers',
            '(Software) Developer, Designer, or Architect': 'AI developers',
            'Data Scientist or Data Analyst': 'AI developers',
            'Information Security Analyst or Engineer': 'Security/Privacy',
            'Information Privacy Analyst or Engineer': 'Security/Privacy',
            'AI Ethicist': 'AI Researcher, AI Ethicist',
            'AI Researcher': 'AI Researcher, AI Ethicist',
            '(Software) Quality Assurance Engineer or Tester': 'QA and Maintenance',
            'Other, please specify:': 'Other',
            'IT': 'Other'
        }
    },
    'Dev Experience': {
        'column': 32,
        'mapping': {
            'None': 'None',
            '1-2 Years': '1-2 Years',
            '2-5 Years': '2-5 Years',
            '5-10 Years': '5-10 Years',
            '10+ Years': '10+ Years'
        }
    },
    'Education': {
        'column': 21,
        'mapping': {
            "High School Degree": "High School Degree",
            "Bachelor's Degree": "Bachelor's Degree",
            "Master's Degree (i.e., MSc., M.A., etc.)": "Graduate Education",
            "MBA (Master of Business Administration)": "Graduate Education",
            "Graduate Certificates": "Graduate Education",
            "Ph.D.": "Ph.D.",
            "Other, please specify": "Other"
        }
    },
    'Company Size': {
        'column': 25,
        'mapping': {
            '1-5 Employees': '1-5 Employees',
            '6-20 Employees': '6-20 Employees',
            '21-50 Employees': '21-50 Employees',
            '51-100 Employees': '51-100 Employees',
            '101+ Employees': '100+ Employees'
        }
    },
    'Gender': {
        'column': 19,
        'mapping': None  # Use raw values for Gender
    }
}


# Demographics and the column number they are found in ai_study_finalized
demographics = {
    'Location': {
        'column': 17,
        'mapping': {
            'north america': 'North America',
            'europe': 'Europe',
            'asia': 'Asia',
        }
    }
}

SPECIAL_PHRASES = [
    "other, please specify", "Master's Degree (i.e., MSc., M.A., etc.)", "Information Science (e.g., IT or MIS)", "Administrative role (CEO, Chief Technical Officer, Chief Operating Officer, Chief Information Officer)", "Scrum Master, Product Manager, or Project Manager", "(Software) Developer, Designer, or Architect", "Chatbots, Personal Assistants or Recommender Systems", "Programming Analysis (e.g., Code Completion or Code Generation)", "Model provider APIs, i.e., OpenAI, Anthropic, DeepInfra, LLama", "RNN, LSTM / GRU", "Clean Data to Remove, Mitigate, or Minimize Biases", "Integrating Privacy Enhancing Technologies (PETs) in AI systems (e.g., differential privacy algorithms, federated learning models, etc.)", "Continuously, as part of an ongoing process", "Professional organizations (e.g., ACM, IEEE, ACL, AAAI, etc.)", 
]




