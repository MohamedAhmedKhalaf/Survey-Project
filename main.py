from turtle import width
from imports import FastAPI, HTTPException, HTMLResponse, px, to_html, pd, sns, Jinja2Templates, Request
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64
import io
import uvicorn
from scipy.stats import chi2_contingency
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import TomekLinks
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from imblearn.under_sampling import TomekLinks
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
import json # Import json for potentially handling classification report as JSON
import numpy as np



app = FastAPI()
templates = Jinja2Templates(directory="templates")


host = "127.0.0.1"
port = 8000



df = pd.read_csv('./cleaned_full_survey_data.csv')
profession_df = pd.read_csv("processed-datasets/profession_categories.csv")
df_infer = df.copy()

@app.get('/')
def root():
    return {'test': 'done'}


@app.get('/descriptive_stats', response_class=HTMLResponse)
async def descriptive_stats(request: Request):
    # Use a copy of the original DataFrame to avoid modifying it globally
    df_stats = df.copy()

    # Define the rating columns to analyze
    rating_columns = [
        'Python_Community_Support',
        'Python_Execution_Speed',
        'Python_Ease_of_Use',
        'Python_Documentation',
        'Python_Concurrency_Features',
        'Python_Readability',
        'Java_Community_Support',
        'Java_Execution_Speed',
        'Java_Ease_of_Use',
        'Java_Documentation',
        'Java_Concurrency_Features',
        'Java_Readability',
        'R_Community_Support',
        'R_Execution_Speed',
        'R_Ease_of_Use',
        'R_Documentation',
        'R_Concurrency_Features',
        'R_Readability'
    ]

    # --- Data Cleaning for Rating Columns ---
    # Replace non-numeric entries with NaN
    for col in rating_columns:
        if col in df_stats.columns: # Check if the column exists
            # Using regex=False for clarity and to avoid potential issues
            df_stats[col] = df_stats[col].replace(['Not Applicable', 'Not Applicable ', 'Not Applicable\n', '.', '..', '...'], np.nan, regex=False)
            # Attempt to convert to numeric, coercion handles errors by turning them to NaN
            df_stats[col] = pd.to_numeric(df_stats[col], errors='coerce')
        else:
            print(f"Warning: Rating column '{col}' not found in DataFrame.")


    # --- Calculate Descriptive Statistics ---
    # Select only the existing rating columns that were successfully converted (or attempted) to numeric
    existing_rating_columns = [col for col in rating_columns if col in df_stats.columns]
    numeric_df_stats = df_stats[existing_rating_columns]

    # Aggregating functions on the DataFrame subset
    # Need to handle the case where a column might become all NaN after cleaning
    descriptive_stats_df = numeric_df_stats.agg(
        ['mean', 'median', 'std', lambda x: x.mode().tolist() if not x.mode().empty else [np.nan]] # mode returns a Series, take first item or list
    )
    descriptive_stats_df = descriptive_stats_df.rename(index={'<lambda>': 'mode'})

    # Transpose the DataFrame so columns are rows for better display in HTML
    descriptive_stats_df_transposed = descriptive_stats_df.T
    descriptive_stats_df_transposed.index.name = ' ' # Name the index column

    # Convert the transposed DataFrame to HTML
    # --- FIX FOR TypeError: 'str' object is not callable ---
    # Provide float_format as a callable lambda function instead of a string
    stats_html = descriptive_stats_df_transposed.to_html(
        classes='styled-table',
        float_format=lambda x: f'{x:.2f}', # Use a lambda function for formatting floats
        na_rep='N/A'         # Represent NaN as 'N/A'
    )
    # --- END OF FIX ---

    # --- Calculate and Visualize Frequency Distributions ---
    plot_data = [] # List to store dictionaries of {'title': title, 'html': plot_html}

    for col in existing_rating_columns:
        # Drop NaN values and check if there are any non-NaN values left
        non_nan_values = df_stats[col].dropna()

        # Ensure data is integer type if appropriate for bar chart categories
        # Only convert if there are values to avoid error on empty series
        if not non_nan_values.empty:
             try:
                 non_nan_values = non_nan_values.astype(int)
             except ValueError:
                 # If conversion fails (e.g., still non-int floats slipped through),
                 # keep as is or handle as needed. For ratings 1-5, int should work.
                 pass


        # Need at least 2 unique values for a meaningful distribution plot
        if not non_nan_values.empty and len(non_nan_values.unique()) > 1:
            # Calculate frequencies
            freq_dist_series = non_nan_values.value_counts().sort_index()
            # Convert the Series to a DataFrame for Plotly
            freq_dist_df = freq_dist_series.reset_index()
            freq_dist_df.columns = [col, 'Count']

            # Create a bar plot using Plotly Express
            fig = px.bar(freq_dist_df,
                         x=col,
                         y='Count',
                         title=f'Distribution of {col}',
                         labels={col: "Rating (1-5)", "Count": "Count"},
                         text='Count') # Display text labels (counts) on the bars

            # Update layout for theme consistency and readability
            fig.update_layout(
                xaxis_tickangle=-45, # Rotate x-axis labels if needed
                xaxis=dict(type='category', tickmode='linear', categoryorder='category ascending'), # Treat x-axis as categorical, force all ticks, sort categories
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                title_font=dict(color='white'),
                margin=dict(l=50, r=50, t=50, b=50) # Adjust margins
            )

            # Convert figure to HTML and add to list
            plot_html = to_html(fig, full_html=False)
            plot_data.append({'title': f'Distribution of {col}', 'html': plot_html}) # Store title and html

        elif not non_nan_values.empty and len(non_nan_values.unique()) == 1:
             # Handle case where there's only one unique value
             plot_data.append({'title': f'Distribution of {col}', 'html': f"<p>Only one value ({non_nan_values.iloc[0]}) observed for this metric.</p>"})
        else:
             # Handle case where column is empty after cleaning
             plot_data.append({'title': f'Distribution of {col}', 'html': "<p>No valid data available for this metric after cleaning.</p>"})


    return templates.TemplateResponse("descriptive_stats.html", {
        "request": request,
        "descriptive_stats_table_html": stats_html,
        "frequency_plot_data": plot_data, # Pass the list of plot data dictionaries
        'host': host,
        'port': port
    })

@app.get('/infer_stat', response_class=HTMLResponse)
def infer_stat(request: Request):

    # Use the copy for modifications within this route
    df_local = df_infer.copy()

    chi_square_test_columns = [
        "What is your age?",
        "What is your gender?",
        "What is your current role?",
        "How many years of programming experience do you have?",
        "Languages_Used",
        "Python_Community_Support",
        "Java_Community_Support",
        "Which language do you use most frequently?",
        "  Which language do you prefer for the following tasks?   [Data Science & Machine Learning]",
        "  Which language do you prefer for the following tasks?   [Web Development]",
        "  Which language do you prefer for the following tasks?   [Mobile App Development]",
        "  Which language do you prefer for the following tasks?   [Enterprise Applications]",
        "  Which language do you prefer for the following tasks?   [Statistical Analysis]",
        "Which language do you perceive as the most efficient for your tasks?",
        "Python_Execution_Speed",
        "Java_Execution_Speed",
        "Python_Ease_of_Use",
        "Java_Ease_of_Use",
        "Python_Documentation",
        "Java_Documentation",
        "Python_Concurrency_Features",
        "Java_Concurrency_Features",
        "Python_Readability",
        "Java_Readability",
        "Which language do you think will dominate the job market in the next 5 years?",
        "Which language was the easiest for you to learn?",
        "Which language has the most beginner-friendly documentation and learning resources?",
        "In your field, which language is the most commonly used?",
        "Which language do you believe is most in demand for jobs?",
        "Which language would you recommend for someone entering your industry?",
        "profession",
        "Have you ever contributed to an open-source project in any of these languages?"
    ]

    # Ensure profession_df has the same index or merge carefully
    # Assuming they align row-wise after initial loading
    # Add a check to ensure both dataframes have the same number of rows
    if len(df_local) != len(profession_df):
         raise HTTPException(status_code=500, detail="Data mismatch: Survey data and profession data have different lengths.")

    df_local['profession'] = profession_df['Predicted_Category']

    # Handle potential KeyError if the column doesn't exist in the original df_local
    if 'What is your current major or job field?' in df_local.columns:
        df_local.drop(columns='What is your current major or job field?', inplace=True)
    else:
         print("Warning: 'What is your current major or job field?' column not found in DataFrame.")


    # Filter for relevant professions
    df_local = df_local[df_local['profession'].isin(['Data Science', 'Software Development'])].copy(deep=True)

    # --- Chi-Square Test Function (same as original) ---
    def chi_tester(df, col1, col2):
        """
        Performs a Chi-Square test for independence on two columns of a DataFrame,
        excluding 'Not Applicable' values.
        Returns the p-value. Returns 1.0 if test cannot be performed.
        """
        # Create contingency table, excluding 'Not Applicable'
        # Use .dropna() before crosstab might be safer if NaNs are possible too
        filtered_df = df[(df[col1].notna()) & (df[col2].notna()) &
                         (df[col1] != 'Not Applicable') & (df[col2] != 'Not Applicable')].copy()


        # Check if filtered_df is empty or results in a table with zero size
        if filtered_df.empty:
            # Cannot perform test if no data remains after filtering
            return 1.0 # Return a high p-value indicating no relationship could be tested

        # Ensure there is variability left after filtering
        # Also check for minimum required data points or table dimensions for chi2_contingency
        if filtered_df[col1].nunique() <= 1 or filtered_df[col2].nunique() <= 1:
            # Cannot perform test if one or both columns have only one unique value
            return 1.0 # Return a high p-value

        try:
            table = pd.crosstab(filtered_df[col1], filtered_df[col2])

            # Check if the crosstab is valid for chi-square (at least 2x2 or larger, with some minimum expected frequencies)
            # chi2_contingency handles many cases, but can fail on tiny or degenerate tables.
            # A simple check for size > 1 can prevent some errors, though not all edge cases.
            if table.shape[0] <= 1 or table.shape[1] <= 1:
                 return 1.0

            # Check for zero row/column sums which can cause errors
            if table.sum(axis=1).min() == 0 or table.sum(axis=0).min() == 0:
                 return 1.0


            chi2, p, dof, expected = chi2_contingency(table)
            # Check if p is NaN, which can happen in rare cases with degenerate data
            if pd.isna(p):
                return 1.0
            return p
        except ValueError as e:
            # This might happen if the resulting table has dimensions that
            # chi2_contingency cannot handle (e.g., 0xN or Nx0 after internal checks)
            print(f"Chi-square test failed for {col1} vs {col2}: {e}")
            return 1.0 # Return a high p-value if test fails
        except Exception as e:
            # Catch any other unexpected errors
            print(f"An unexpected error occurred during chi-square test for {col1} vs {col2}: {e}")
            return 1.0


    # --- Columns to Test (same as original) ---
    # Ensure all these columns actually exist in the filtered df_local
    available_columns = [col for col in chi_square_test_columns if col in df_local.columns]

    # --- Perform Chi-Square Tests and Collect Significant Pairs ---
    # Collect plot HTMLs and pair info
    significant_results = []
    plot_htmls = []

    tested_pairs_count = 0

    # Use the filtered list of available columns for testing
    for i, col1 in enumerate(available_columns):
        for col2 in available_columns[i+1:]:
            tested_pairs_count += 1
            p_value = chi_tester(df_local, col1, col2)

            if p_value is not None and p_value < 0.05:
                significant_results.append({'col1': col1, 'col2': col2, 'p': p_value})

                # Create the crosstab table for visualization (using df_local for consistency with filtering)
                crosstab = pd.crosstab(df_local[col1], df_local[col2])

                # Skip plotting if the crosstab is empty after filtering, though chi_tester should handle this
                if crosstab.empty:
                     continue

                # Create the heatmap figure
                fig_infer = go.Figure(data=go.Heatmap(
                        z=crosstab.values,
                        x=crosstab.columns.tolist(),
                        y=crosstab.index.tolist(),
                        colorscale='Viridis',
                        text=crosstab.values,
                        texttemplate="%{text}",
                        hovertemplate='<b>%{x}</b><br><b>%{y}</b><br>Count: %{z}<extra></extra>'
                    ))

                # Update layout
                fig_infer.update_layout(
                    title=f"Crosstab Heatmap: '{col1}' vs '{col2}'<br>(p={p_value:.4f})",
                    xaxis_title=col2,
                    yaxis_title=col1,
                    xaxis={'side': 'bottom'},
                    margin=dict(l=100, r=100, t=100, b=100),
                    plot_bgcolor='rgba(0,0,0,0)', # Match theme background
                    paper_bgcolor='rgba(0,0,0,0)', # Match theme background
                    font=dict(color='white'), # Match theme text color
                    title_font=dict(color='white'),
                    # xaxis=dict(tickfont=dict(color='white')),
                    # yaxis=dict(tickfont=dict(color='white'))
                )

                # Convert figure to HTML and add to list
                plot_html = to_html(fig_infer, full_html=False)
                plot_htmls.append(plot_html)


    return templates.TemplateResponse("infer.html", {
        "request": request,
        "significant_results": significant_results, # Pass the list of significant pairs and p-values
        "plot_htmls": plot_htmls,                 # Pass the list of plot HTML strings
        'host': host,
        'port': port
    })







@app.get('/dashboard', response_class=HTMLResponse)
async def get_gender_portion(request: Request):
    # making a dataframe that has the gender and the freq (PIE CHART)
    genders_Series = df[['What is your gender?']].value_counts()
    genders_df = genders_Series.to_frame()
    genders_df = genders_df.reset_index()

    # AGE DIST (BAR-CHART)
    age_Series = df['What is your age?'].value_counts().reset_index()
    age_Series.columns = ['Age Category', 'count']

    # Roles Dist 
    roles_Series = df[['What is your current role?']].value_counts().reset_index()
    roles_Series.columns = ['Role', 'count']

    # Years of Exp
    exp = df[['How many years of programming experience do you have?']].value_counts().reset_index()
    exp.columns = ['exp','count']


    # Major
    major = df[['What is your current major or job field?']]
    major_keywords = {
    "data science": "data",
    "data scientist": "data",
    "software engineering": "software",
    "software engineer": "software",
    "software development": "software",
    "developer": "software",
    "programmer": "software",
    "web development": "software",
    "frontend": "software",
    "backend": "software",
    "AI specialist": "ai",
    "machine learning": "ai",
    "cyber security": "security",
    "network engineer": "networking",
    "computer science": "software",
    "math teacher": "education",
    "teacher": "education",
    "engineer": "engineering"
    }
    
    major_keywords = {k.lower(): v for k, v in major_keywords.items()}

    def classify_role(text):
        text = str(text).lower()
        for key in major_keywords:
            if key in text:
                return major_keywords[key]
        return "other"  

    major['major'] = major['What is your current major or job field?'].apply(classify_role)

    major_Series = major[['major']].value_counts()
    major_df = major_Series.to_frame()
    major_df = major_df.reset_index()


    # Langs
    lang_used = df[['Languages_Used']].value_counts().reset_index()


    # gender and exp
    gender_exp = df[['What is your gender?', 'How many years of programming experience do you have?',]]
    gender_exp.columns = ['Gender','Experience']


    # Job and Lang
    job_lang = df[['What is your current role?','Languages_Used']]
    job_lang.columns = ['Role','Lang']
    lang_counts = job_lang.groupby(['Role', 'Lang']).size().reset_index(name='Count')
    lang_pivot = lang_counts.pivot(index='Role', columns='Lang', values='Count').fillna(0)
    lang_pivot = lang_pivot.reset_index()
    lang_long = lang_pivot.melt(id_vars='Role', var_name='Language', value_name='Count')


    # Role and Exp

    role_exp = df[['What is your current role?', 'How many years of programming experience do you have?']]
    role_exp.columns = ['Role', 'Exp']
    def parse_experience(exp):
        if pd.isna(exp):
            return None
        exp = exp.strip().lower().replace("years", "").strip()
        
        if '+' in exp:
            return float(exp.replace('+', '').strip())
        elif '-' in exp:
            parts = exp.split('-')
            return (float(parts[0]) + float(parts[1])) / 2
        else:
            try:
                return float(exp)
            except:
                return None

    role_exp['Exp_numeric'] = role_exp['Exp'].apply(parse_experience)

    avg_exp_by_role = role_exp.groupby('Role')['Exp_numeric'].mean().reset_index()


    #--------------------------------------------------------------------------------------------------------

    # pie chart with plotly
    fig = px.pie(genders_df, names='What is your gender?', values='count', template='plotly_white')
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  # Set all text (title, ticks) to white
        title_font=dict(color='white'),  # Set the title text color to white
        xaxis=dict(tickfont=dict(color='white')),  # Set x-axis tick labels to white
        yaxis=dict(tickfont=dict(color='white'))  # Set y-axis tick labels to white
    )

    # bar chart 
    fig_Age_Dist = px.bar(age_Series,
                          x='Age Category',
                          y='count',
                          labels={'Age Category': 'Age Category', 'count': 'Frequency'},
                          )
    fig_Age_Dist.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  # Set all text (title, ticks) to white
        title_font=dict(color='white'),  # Set the title text color to white
        xaxis=dict(tickfont=dict(color='white')),  # Set x-axis tick labels to white
        yaxis=dict(tickfont=dict(color='white'))  # Set y-axis tick labels to white
    )

    # horizontal bar chart For Roles Dist
    fig_roles = px.bar(data_frame=roles_Series, x='count', y='Role', orientation='h')
    fig_roles.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  # Set all text (title, ticks) to white
        title_font=dict(color='white'),  # Set the title text color to white
        xaxis=dict(tickfont=dict(color='white')),  # Set x-axis tick labels to white
        yaxis=dict(tickfont=dict(color='white'))  # Set y-axis tick labels to white
    )

    # Years of Experience Distribution
    fig_years_exp = px.bar(exp, 
             x='exp', 
             y='count',
             labels={'exp': 'Programming Experience', 'count': 'Count'}
            )

    fig_years_exp.update_layout(
        xaxis_tickangle=-15,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  # Set all text (title, ticks) to white
        title_font=dict(color='white'),  # Set the title text color to white
        xaxis=dict(tickfont=dict(color='white')),  # Set x-axis tick labels to white
        yaxis=dict(tickfont=dict(color='white'))  # Set y-axis tick labels to white
    )

    # Major 
    fig_major = px.pie(major_df,names= 'major',values= 'count')
    fig_major.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  
        title_font=dict(color='white'),  
        xaxis=dict(tickfont=dict(color='white')), 
        yaxis=dict(tickfont=dict(color='white'))  
    )

    # Langs 

    fig_langs = px.pie(lang_used,names='Languages_Used', values='count')
    fig_langs.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),  
    title_font=dict(color='white'),  
    xaxis=dict(tickfont=dict(color='white')), 
    yaxis=dict(tickfont=dict(color='white'))  
    )

    # gender and exp
    fig_gender_exp = px.box(
        gender_exp,
        x='Gender',
        y='Experience',
        color='Gender',
    )
    fig_gender_exp.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  
        title_font=dict(color='white'),  
        xaxis=dict(tickfont=dict(color='white')), 
        yaxis=dict(tickfont=dict(color='white'))  
    )

    # Job and Lang
    fig_job_lang = px.bar(
    lang_long,
        x='Role',
        y='Count',
        color='Language',
        title='Most Used Programming Languages by Role',
        labels={'Count': 'Number of Mentions', 'Role': 'Job Role'},
    )

    fig_job_lang.update_layout(
        barmode='stack',
        xaxis_tickangle=-45,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  
        title_font=dict(color='white'),  
        xaxis=dict(tickfont=dict(color='white')), 
        yaxis=dict(tickfont=dict(color='white')) 

    )

    # Role and Exp
    fig_role_exp = px.bar(
        avg_exp_by_role,
        x='Role',
        y='Exp_numeric',
        title='Average Programming Experience by Role',
        labels={'Role': 'Job Role', 'Exp_numeric': 'Avg. Years of Experience'}
    )

    fig_role_exp.update_layout(
        xaxis_tickangle=-45,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  
        title_font=dict(color='white'),  
        xaxis=dict(tickfont=dict(color='white')), 
        yaxis=dict(tickfont=dict(color='white')) 

    )


    #-----------------------------------------------------------------------------

    # convert the plots to html 
    plot_div = to_html(fig, full_html=False)
    age_dist_div = to_html(fig_Age_Dist, full_html=False)
    roles_dist_div = to_html(fig_roles, full_html=False)
    exp_dist_div = to_html(fig_years_exp,full_html= False)
    major_dist_div = to_html(fig_major,full_html=False)
    langs_dist_div = to_html(fig_langs,full_html=False)
    gender_exp_div = to_html(fig_gender_exp,full_html=False)
    job_lang_div = to_html(fig_job_lang,full_html=False)
    role_exp_div = to_html(fig_role_exp,full_html=False)


    return templates.TemplateResponse("respondents.html", {
        "request": request, 
        "plot_div": plot_div,
        "age_dist_div": age_dist_div,
        'roles_dist_div': roles_dist_div,
        'exp_dist_div': exp_dist_div,
        'major_dist_div':major_dist_div,
        'langs_dist_div':langs_dist_div,
        'gender_exp_div':gender_exp_div,
        'job_lang_div':job_lang_div,
        'role_exp_div':role_exp_div,
        'host':host,
        'port':port
    })






@app.get('/PythonVsJava', response_class=HTMLResponse)
async def get_gender_portion(request: Request):




    # Dealing with data and logic

    # fix added 
    major_keywords = {
    "data science": "data",
    "data scientist": "data",
    "software engineering": "software",
    "software engineer": "software",
    "software development": "software",
    "developer": "software",
    "programmer": "software",
    "web development": "software",
    "frontend": "software",
    "backend": "software",
    "AI specialist": "ai",
    "machine learning": "ai",
    "cyber security": "security",
    "network engineer": "networking",
    "computer science": "software",
    "math teacher": "education",
    "teacher": "education",
    "engineer": "engineering"
    }

    major_keywords = {k.lower(): v for k, v in major_keywords.items()}

    def classify_role(text):
        text = str(text).lower()
        for key in major_keywords:
            if key in text:
                return major_keywords[key]
        return "other"  

    df['general_role'] = df['What is your current major or job field?'].apply(classify_role)

    # Languages used 
    lang_used = df[['Which language do you use most frequently?']].value_counts().reset_index()
    
    # Langs for DS
    lang_used_ds = df[['  Which language do you prefer for the following tasks?   [Data Science & Machine Learning]']].value_counts().reset_index()
    lang_used_ds.columns = ['lang','count']


    # Langs for Web
    lang_used_web = df[['  Which language do you prefer for the following tasks?   [Web Development]']].value_counts().reset_index()
    lang_used_web.columns = ['lang','count']

    # Langs For Entrprice Apps
    lang_used_entrprice= df[['  Which language do you prefer for the following tasks?   [Enterprise Applications]']].value_counts().reset_index()
    lang_used_entrprice.columns = ['lang','count']

    # Langs For Stat
    lang_used_stat = df[['  Which language do you prefer for the following tasks?   [Statistical Analysis]']].value_counts().reset_index()
    lang_used_stat.columns = ['lang','count']

    # Switch Between langs
    switch_between_langs = df[['How often do you switch between these languages?']].value_counts().reset_index()
    switch_between_langs.columns = ['lang','count']

    # Most Effiecient for ur task
    task = df[['Which language do you perceive as the most efficient for your tasks?']].value_counts().reset_index()
    task.columns = ['lang','count']

    # future 
    future = df[['Which language do you think will dominate the job market in the next 5 years?']].value_counts().reset_index()
    future.columns = ['lang','count']

    # easier
    easiest = df[['Which language was the easiest for you to learn?']].value_counts().reset_index()
    easiest.columns = ['lang','count']

    # docs and resources
    resourses = df[['Which language has the most beginner-friendly documentation and learning resources?']].value_counts().reset_index()
    resourses.columns = ['lang','count']

    # contribution 
    contributing = df[['Have you ever contributed to an open-source project in any of these languages?']].value_counts().reset_index()
    contributing.columns = ['lang','count']

    # langs vs gender No Need  
    # Age vs lang No Need
    # exp vs lang no need
    # langs in roles no need
    
    # prefered vs entrprice apps
    source_labels = df['Which language do you use most frequently?']
    target_labels = df['  Which language do you prefer for the following tasks?   [Enterprise Applications]']

    label_list = list(set(source_labels) | set(target_labels))
    label_dict = {label: i for i, label in enumerate(label_list)}

    df_sankey = pd.DataFrame({
        'source': source_labels.map(label_dict),
        'target': target_labels.map(label_dict)
    })
    sankey_data = df_sankey.value_counts().reset_index(name="count")

    # summary
    cols = [
    '  Which language do you prefer for the following tasks?   [Data Science & Machine Learning]',
    '  Which language do you prefer for the following tasks?   [Web Development]',
    '  Which language do you prefer for the following tasks?   [Mobile App Development]',
    '  Which language do you prefer for the following tasks?   [Enterprise Applications]',
    '  Which language do you prefer for the following tasks?   [Statistical Analysis]'
    ]

    lang_count = {}
    for col in cols:
        for lang in ['Python', 'Java']:
            lang_count.setdefault(lang, []).append(df[col].value_counts().get(lang, 0))


    # Tree Map No need

    # word cloud
    text = " ".join(df["If you could only use one of these three languages for all future projects, which would you choose and why?"].dropna())

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # violin no need 
    
    # funnel Adoption py and java
    funnel_df = pd.DataFrame({
    "Stage": ["Perceived Future Dominance", "Most Used", "Recommended for Entry"],
    "Python": [
        (df['Which language do you think will dominate the job market in the next 5 years?'] == 'Python').sum(),
        (df['Which language do you use most frequently?'] == 'Python').sum(),
        (df['Which language would you recommend for someone entering your industry?'] == 'Python').sum()
    ],
    "Java": [
        (df['Which language do you think will dominate the job market in the next 5 years?'] == 'Java').sum(),
        (df['Which language do you use most frequently?'] == 'Java').sum(),
        (df['Which language would you recommend for someone entering your industry?'] == 'Java').sum()
    ]
    })

    # contrib and langs
    oss_contrib_col = 'Have you ever contributed to an open-source project in any of these languages?'

    plot_df = df[(df['Which language do you use most frequently?'] != 'Unknown') & (df[oss_contrib_col] != 'Unknown')].copy()

    plot_df[oss_contrib_col] = plot_df[oss_contrib_col].apply(lambda x: 'Yes' if x == 'Yes' else ('No' if x == 'No' else 'Other/Unknown'))

    # bar1 
    speed_df = df[['Python_Execution_Speed', 'R_Execution_Speed', 'Java_Execution_Speed']].replace('Not Applicable', np.nan)
    speed_df = speed_df.apply(pd.to_numeric, errors='coerce')
    avg_speeds = speed_df.mean().reset_index()
    avg_speeds.columns = ['Language', 'Average_Speed']

    # bar2 
    ease_df = df[['Python_Ease_of_Use', 'R_Ease_of_Use', 'Java_Ease_of_Use']]
    ease_df = ease_df.apply(pd.to_numeric, errors='coerce')
    avg_ease = ease_df.mean().reset_index()
    avg_ease.columns = ['Language', 'Average_Ease']

    # bar3 
    doc_df = df[['Python_Documentation', 'R_Documentation', 'Java_Documentation']]
    doc_df = doc_df.apply(pd.to_numeric, errors='coerce')
    avg_doc = doc_df.mean().reset_index()
    avg_doc.columns = ['Language', 'Average_Doc']

    # bar4
    # Select and clean concurrency columns
    concurrency_df = df[['Python_Concurrency_Features', 'R_Concurrency_Features', 'Java_Concurrency_Features']]
    concurrency_df = concurrency_df.apply(pd.to_numeric, errors='coerce')

    # Calculate average
    avg_concurrency = concurrency_df.mean().reset_index()
    avg_concurrency.columns = ['Language', 'Average_Concurrency']

    # bar5
    # Select and clean readability columns
    readability_df = df[['Python_Readability', 'R_Readability', 'Java_Readability']]
    readability_df = readability_df.apply(pd.to_numeric, errors='coerce')

    # Calculate average
    avg_readability = readability_df.mean().reset_index()
    avg_readability.columns = ['Language', 'Average_Readability']


    #-------------------------------------------------------

    # Plots 

    # langs 
    fig_langs_used = px.pie(lang_used,names='Which language do you use most frequently?', values='count')
    fig_langs_used.update_layout(
        xaxis_tickangle=-45,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  
        title_font=dict(color='white'),  
        xaxis=dict(tickfont=dict(color='white')), 
        yaxis=dict(tickfont=dict(color='white')) 
    )


    # Langs for Data 
    fig_langs_data = px.pie(lang_used_ds,names='lang', values='count')
    fig_langs_data.update_layout(
        xaxis_tickangle=-45,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  
        title_font=dict(color='white'),  
        xaxis=dict(tickfont=dict(color='white')), 
        yaxis=dict(tickfont=dict(color='white')) 
    )

    # Langs For Web
    fig_langs_web = px.pie(lang_used_web,names='lang', values='count')
    fig_langs_web.update_layout(
        xaxis_tickangle=-45,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  
        title_font=dict(color='white'),  
        xaxis=dict(tickfont=dict(color='white')), 
        yaxis=dict(tickfont=dict(color='white')) 
    )

    # Langs For Entrprice Apps
    fig_langs_entr = px.pie(lang_used_entrprice,names='lang', values='count')
    fig_langs_entr.update_layout(
        xaxis_tickangle=-45,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  
        title_font=dict(color='white'),  
        xaxis=dict(tickfont=dict(color='white')), 
        yaxis=dict(tickfont=dict(color='white')) 
    )

    # Langs for Stat
    fig_langs_stat = px.pie(lang_used_stat,names='lang', values='count')
    fig_langs_stat.update_layout(
        xaxis_tickangle=-45,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  
        title_font=dict(color='white'),  
        xaxis=dict(tickfont=dict(color='white')), 
        yaxis=dict(tickfont=dict(color='white')) 
    )

    # Switch between langs
    fig_switch_langs = px.bar(switch_between_langs, 
             x='lang', 
             y='count',
             title='switch between these languages',
             labels={'lang': 'from 1 to 5', 'count': 'Count'}
             )
    fig_switch_langs.update_layout(
        xaxis_tickangle=-15,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  
        title_font=dict(color='white'),  
        xaxis=dict(tickfont=dict(color='white')), 
        yaxis=dict(tickfont=dict(color='white')) 
    )

    # most effiecient for ur task
    fig_most_eff = px.bar(task, 
             x='lang', 
             y='count',
             title='Most Efficient For Your Tasks',
             labels={'lang': '', 'count': 'Count'}
             )
    fig_most_eff.update_layout(
        xaxis_tickangle=-15,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  
        title_font=dict(color='white'),  
        xaxis=dict(tickfont=dict(color='white')), 
        yaxis=dict(tickfont=dict(color='white')) 

    )

    # future
    
    fig_future = px.bar(future, 
             x='lang', 
             y='count',
             title='Future Dominace In The Next 5 Years ',
             labels={'lang': '', 'count': 'Count'}
    )
    fig_future.update_layout(
        xaxis_tickangle=-15,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  
        title_font=dict(color='white'),  
        xaxis=dict(tickfont=dict(color='white')), 
        yaxis=dict(tickfont=dict(color='white'))
    )

    # easier 
    fig_easier = px.pie(easiest,names='lang', values='count')
    fig_easier.update_layout(
        xaxis_tickangle=-15,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  
        title_font=dict(color='white'),  
        xaxis=dict(tickfont=dict(color='white')), 
        yaxis=dict(tickfont=dict(color='white'))
    )

    # docs and resources 
    fig_doc_resources = px.pie(resourses,names='lang', values='count')
    fig_doc_resources.update_layout(
        xaxis_tickangle=-15,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  
        title_font=dict(color='white'),  
        xaxis=dict(tickfont=dict(color='white')), 
        yaxis=dict(tickfont=dict(color='white'))
    )

    # contribution 
    fig_contribution = px.pie(contributing,names='lang', values='count')
    fig_contribution.update_layout(
        xaxis_tickangle=-15,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  
        title_font=dict(color='white'),  
        xaxis=dict(tickfont=dict(color='white')), 
        yaxis=dict(tickfont=dict(color='white'))
    )

    # lang vs gender
    gender_language_fig = px.histogram(df, x='Which language do you use most frequently?', color='What is your gender?', barmode='group',
                                title='Gender vs. Most Frequent Language')
    gender_language_fig.update_layout(
        xaxis_tickangle=-15,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  
        title_font=dict(color='white'),  
        xaxis=dict(tickfont=dict(color='white')), 
        yaxis=dict(tickfont=dict(color='white'))
    )
    
    # age vs lang
    fig_age_lang = px.histogram(df, x='What is your age?', color='Which language do you use most frequently?',
                   title='Age Distribution by Primary Language',
                   category_orders={'Which language do you use most frequently?': ['Python', 'Java', 'R']})
    fig_age_lang.update_layout(
        barmode='group',
        xaxis_tickangle=-15,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  
        title_font=dict(color='white'),  
        xaxis=dict(tickfont=dict(color='white')), 
        yaxis=dict(tickfont=dict(color='white'))

    )

    # exp vs lang
    fig_exp_lang = px.box(df, x='Which language do you use most frequently?', y='How many years of programming experience do you have?',
             title='Years Vs Land',
             color='Which language do you use most frequently?')
    fig_exp_lang.update_yaxes(type='category')
    fig_exp_lang.update_layout(
        xaxis_tickangle=-15,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  
        title_font=dict(color='white'),  
        xaxis=dict(tickfont=dict(color='white')), 
        yaxis=dict(tickfont=dict(color='white'))
    )

    # langs for roles
    fig_langs_roles = px.sunburst(df, path=['general_role', 'Which language do you use most frequently?'],
                  title='Language Usage by General Role')
    fig_langs_roles.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  
        title_font=dict(color='white'),  
        xaxis=dict(tickfont=dict(color='white')), 
        yaxis=dict(tickfont=dict(color='white'))
    )

    # prefered vs entrprice apps 

    fig_per_entr = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=label_list
    ),
    link=dict(
        source=sankey_data['source'],
        target=sankey_data['target'],
        value=sankey_data['count']
    ))])

    fig_per_entr.update_layout(
        title_text="Language Transition: Use → Enterprise Preference",
        font_size=10,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  
        title_font=dict(color='white'),  
        xaxis=dict(tickfont=dict(color='white')), 
        yaxis=dict(tickfont=dict(color='white'))

    )

    # summary
    fig_summary = go.Figure()
    for lang in ['Python', 'Java']:
        fig_summary.add_trace(go.Bar(
            x=[c.split("[")[-1].replace("]", "") for c in cols],
            y=lang_count[lang],
            name=lang
        ))

    fig_summary.update_layout(
        barmode='stack',
        title="Language Preference by Task",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  
        title_font=dict(color='white'),  
        xaxis=dict(tickfont=dict(color='white')), 
        yaxis=dict(tickfont=dict(color='white'))

    )

    # tree map
    fig_tree_map = px.treemap(
    df,
    path=['general_role', 'Which language do you use most frequently?'],
    title='Treemap: Most Used Language by General Role',
    color='Which language do you use most frequently?'
    )
    fig_tree_map.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  
        title_font=dict(color='white'),  
        xaxis=dict(tickfont=dict(color='white')), 
        yaxis=dict(tickfont=dict(color='white'))
    )
    
    # word cloud 
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title("Word Cloud: Language Preference Reasons")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    # violin 
    fig_violin = px.violin(
        df,
        y='How many years of programming experience do you have?',
        x='Which language do you use most frequently?',
        box=True,
        points='all',
        title="Experience Distribution by Preferred Language"
    )
    fig_violin.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  
        title_font=dict(color='white'),  
        xaxis=dict(tickfont=dict(color='white')), 
        yaxis=dict(tickfont=dict(color='white'))
    )

    # funnel adoption for py
    fig_funn_py = px.funnel(funnel_df, x='Python', y='Stage', title="Python Adoption Funnel")
    fig_funn_py.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  
        title_font=dict(color='white'),  
        xaxis=dict(tickfont=dict(color='white')), 
        yaxis=dict(tickfont=dict(color='white'))
    )

    # funnel adoption for java
    fig_funn_java = px.funnel(funnel_df, x='Java', y='Stage', title="Java Adoption Funnel")
    fig_funn_java.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  
        title_font=dict(color='white'),  
        xaxis=dict(tickfont=dict(color='white')), 
        yaxis=dict(tickfont=dict(color='white'))
    )
    
    # contrib and langs
    fig_contr_langs = px.bar(plot_df, x='Which language do you use most frequently?', color=oss_contrib_col,
                title='Open Source Contribution Status by Most Frequent Language Used',
                labels={'Which language do you use most frequently?': 'Most Frequent Language',
                        oss_contrib_col: 'Contributed to Open Source'},
                barmode='stack')
    fig_contr_langs.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  
        title_font=dict(color='white'),  
        xaxis=dict(tickfont=dict(color='white')), 
        yaxis=dict(tickfont=dict(color='white'))
    )
    
    # age and effi
    fig_age_effi = px.scatter(
    df,
    x='How many years of programming experience do you have?',
    y='How often do you switch between these languages?',
    size='How often do you seek help from online resources (e.g., Stack Overflow, official documentation)?',
    color='Which language do you perceive as the most efficient for your tasks?',
    animation_frame='What is your age?',
    hover_data=['general_role', 'What is your current major or job field?'],
    title='Animated Bubble Chart: Experience vs. Language Efficiency by Age'
    )
    fig_age_effi.update_layout(
    xaxis_title='Years of Programming Experience',
    yaxis_title='Language Switching Frequency',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),  
    title_font=dict(color='white'),  
    xaxis=dict(tickfont=dict(color='white')), 
    yaxis=dict(tickfont=dict(color='white')),
    width=1000,
    
    )

    # bar1 
    fig_bar1 = px.bar(
    avg_speeds,
    x='Language',
    y='Average_Speed',
    color='Language',
    title='Average Execution Speed by Programming Language',
    text='Average_Speed',
    )

    fig_bar1.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_bar1.update_layout(
        yaxis=dict(range=[0,6]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  
        title_font=dict(color='white'),  
        # xaxis=dict(tickfont=dict(color='white')), 
        # yaxis=dict(tickfont=dict(color='white'))               
    
    )

    # bar2
    fig_bar2 = px.bar(
    avg_ease,
    x='Language',
    y='Average_Ease',
    color='Language',
    title='Average Ease Of Use by Programming Language',
    text='Average_Ease',
    )

    fig_bar2.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_bar2.update_layout(
        yaxis=dict(range=[0,6]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  
        title_font=dict(color='white'),  
        # xaxis=dict(tickfont=dict(color='white')), 
        # yaxis=dict(tickfont=dict(color='white'))               
    )

    # bar3
    fig_bar3 = px.bar(
    avg_doc,
    x='Language',
    y='Average_Doc',
    color='Language',
    title='Average Documentation by Programming Language',
    text='Average_Doc',
    )

    fig_bar3.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_bar3.update_layout(
        yaxis=dict(range=[0,6]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  
        title_font=dict(color='white'),  
        # xaxis=dict(tickfont=dict(color='white')), 
        # yaxis=dict(tickfont=dict(color='white'))               
    )
    # bar4
    fig_bar4 = px.bar(
    avg_concurrency,
    x='Language',
    y='Average_Concurrency',
    color='Language',
    title='Average Concurrency Features by Programming Language',
    text='Average_Concurrency',
)

    fig_bar4.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_bar4.update_layout(
        yaxis=dict(range=[0,6]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  
        title_font=dict(color='white'),  
        # xaxis=dict(tickfont=dict(color='white')), 
        # yaxis=dict(tickfont=dict(color='white'))               
    )

    # bar5 
    fig_bar5 = px.bar(
    avg_readability,
    x='Language',
    y='Average_Readability',
    color='Language',
    title='Average Readability by Programming Language',
    text='Average_Readability',
)

    fig_bar5.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_bar5.update_layout(
        yaxis=dict(range=[0,6]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  
        title_font=dict(color='white'),  
        # xaxis=dict(tickfont=dict(color='white')), 
        # yaxis=dict(tickfont=dict(color='white'))               
    )


    #-------------------------------------------------------


    # convert plots to html 
    lang_used_div = to_html(fig_langs_used,full_html=False)
    lang_used_data_div = to_html(fig_langs_data,full_html=False)
    lang_used_web_div = to_html(fig_langs_web,full_html=False)
    lang_used_entr_div = to_html(fig_langs_entr,full_html=False)
    lang_used_stat_div = to_html(fig_langs_stat,full_html=False)
    switch_langs_div = to_html(fig_switch_langs,full_html=False)
    most_Effi_div = to_html(fig_most_eff,full_html=False)
    future_div = to_html(fig_future,full_html=False)
    easier_div = to_html(fig_easier,full_html=False)
    docs_rosources_div = to_html(fig_doc_resources,full_html=False)
    contribution_div = to_html(fig_contribution,full_html=False)
    gender_lang_div = to_html(gender_language_fig,full_html=False)
    age_lang_div = to_html(fig_age_lang,full_html=False)
    exp_lang_div = to_html(fig_exp_lang,full_html=False)
    langs_roles_div = to_html(fig_langs_roles,full_html=False)
    per_entr_div = to_html(fig_per_entr,full_html=False)
    summary_div = to_html(fig_summary,full_html=False)
    tree_map_div = to_html(fig_tree_map,full_html=False)
    tree_map_div = to_html(fig_tree_map,full_html=False)
    violin_div = to_html(fig_violin,full_html=False)
    funn_py_div = to_html(fig_funn_py,full_html=False)
    funn_java_div = to_html(fig_funn_java,full_html=False)
    contr_langs_div = to_html(fig_contr_langs,full_html=False)
    age_effi_div = to_html(fig_age_effi,full_html=False)
    bar1_div = to_html(fig_bar1,full_html=False)
    bar2_div = to_html(fig_bar2,full_html=False)
    bar3_div = to_html(fig_bar3,full_html=False)
    bar4_div = to_html(fig_bar4,full_html=False)
    bar5_div = to_html(fig_bar5,full_html=False)
    




    return templates.TemplateResponse("PythonVsJava.html", {
        "request": request,
        'lang_used_div':lang_used_div,
        'lang_used_data_div':lang_used_data_div,
        'lang_used_web_div':lang_used_web_div,
        'lang_used_entr_div':lang_used_entr_div,
        'lang_used_stat_div':lang_used_stat_div,
        'switch_langs_div':switch_langs_div,
        'most_Effi_div':most_Effi_div,
        'future_div':future_div,
        'easier_div':easier_div,
        'docs_rosources_div':docs_rosources_div,
        'contribution_div':contribution_div,
        'gender_lang_div':gender_lang_div,
        'age_lang_div':age_lang_div,
        'exp_lang_div':exp_lang_div,
        'langs_roles_div':langs_roles_div,
        'per_entr_div':per_entr_div,
        'summary_div':summary_div,
        'tree_map_div':tree_map_div,
        "wordcloud_image": f"data:image/png;base64,{img_base64}",
        'violin_div':violin_div,
        'funn_py_div':funn_py_div,
        'funn_java_div':funn_java_div,
        'contr_langs_div':contr_langs_div,
        'age_effi_div':age_effi_div,
        'bar1_div':bar1_div,
        'bar2_div':bar2_div,
        "bar3_div":bar3_div,
        'bar4_div':bar4_div,
        "bar5_div":bar5_div,
        'host':host,
        'port':port,

    })




def shorten_feature_name(name):
    """
    Shortens feature names for better readability on plots.
    Handles original names, and attempts to handle OHE suffixes.
    """
    # Start with specific, longer phrases
    name = name.replace('Which language do you use most frequently?', 'Most Frequent Lang')
    name = name.replace('Which language do you prefer for the following tasks?', 'Prefer Task')
    name = name.replace('Which language do you perceive as the most efficient for your tasks?', 'Eff. Lang')
    name = name.replace('Which language do you think will dominate the job market in the next 5 years?', 'Future Dominance')
    name = name.replace('Which language was the easiest for you to learn?', 'Easiest to Learn')
    name = name.replace('Which language has the most beginner-friendly documentation and learning resources?', 'Best Docs/Resources')
    name = name.replace('In your field, which language is the most commonly used?', 'Most Used in Field')
    name = name.replace('Which language do you believe is most in demand for jobs?', 'Most In Demand Jobs')
    name = name.replace('Which language would you recommend for someone entering your industry?', 'Rec. for Entry')
    name = name.replace('Have you ever contributed to an open-source project in any of these languages?', 'OSS Contribution')
    name = name.replace('How often do you switch between these languages?', 'Switch Frequency')
    name = name.replace('How many years of programming experience do you have?', 'Years Exp')
    name = name.replace('What is your age?', 'Age')
    name = name.replace('How often do you seek help from online resources (e.g., Stack Overflow, official documentation)?', 'Seek Help Frequency')

    # Handle language-specific ratings
    name = name.replace('_Community_Support', ' Comm. Support')
    name = name.replace('_Execution_Speed', ' Speed')
    name = name.replace('_Ease_of_Use', ' Ease')
    name = name.replace('_Documentation', ' Docs')
    name = name.replace('_Concurrency_Features', ' Concurrency')
    name = name.replace('_Readability', ' Readability')

    # Handle task brackets (run after 'Prefer Task')
    name = name.replace('   [Data Science & Machine Learning]', ' (DS/ML)')
    name = name.replace('   [Web Development]', ' (Web Dev)')
    name = name.replace('   [Mobile App Development]', ' (Mobile Dev)')
    name = name.replace('   [Enterprise Applications]', ' (Enterprise)')
    name = name.replace('   [Statistical Analysis]', ' (Stats)')

    # Handle OHE suffixes (common patterns) - Order matters!
    # Handle specific responses first, then general language/boolean suffixes
    name = name.replace('_All equally', ' (All Equal)')
    name = name.replace('_It depends', ' (Depends)')
    name = name.replace('_No preference', ' (No Pref)')
    name = name.replace('_Planning to', ' (Plan to)') # For OSS contribution

    # General Language/Boolean suffixes
    name = name.replace('_Python', ' (Py)')
    name = name.replace('_Java', ' (Java)')
    name = name.replace('_R', ' (R)')
    name = name.replace('_Yes', ' (Yes)') # For Yes/No questions like OSS contrib
    name = name.replace('_No', ' (No)')
    name = name.replace('_Male', ' (Male)') # For Gender
    name = name.replace('_Female', ' (Female)')
    name = name.replace('_Non-binary', ' (Non-binary)')

    # Handle profession OHE
    name = name.replace('profession_Data Science', 'Role (DS)')
    name = name.replace('profession_Software Development', 'Role (SWD)')

    # Handle remaining potential OHE suffixes or cleanups
    name = name.replace('Languages_Used_', 'Used: ') # For Languages_Used OHE

    # Remove any lingering special characters or multiple spaces
    name = name.replace('[', '').replace(']', '').replace('?', '').replace('.', '').replace(',', '').strip()
    name = ' '.join(name.split()) # Replace multiple spaces with single space

    return name

# Assume necessary imports and global dataframes/functions (like shorten_feature_name, label_encode_with_exceptions) are available

@app.get('/correlation', response_class=HTMLResponse)
async def get_correlation_matrix(request: Request):
    # Use a copy of the original DataFrame for processing
    df_corr = df_original.copy()

    # --- Data Preprocessing Steps (Replicate necessary ones) ---

    # Add the predicted profession category
    if len(df_corr) != len(profession_df):
         print("Warning: Survey data and profession data have different lengths. Skipping profession merge for correlation.")
         # Decide how to proceed: maybe just drop the profession column or handle the merge carefully
         # For now, if mismatch, don't add profession to df_corr
         profession_column_available = False
    else:
         df_corr['profession'] = profession_df['Predicted_Category']
         # Drop the original 'What is your current major or job field?' if it exists
         if 'What is your current major or job field?' in df_corr.columns:
             df_corr.drop(columns='What is your current major or job field?', inplace=True, errors='ignore')
         profession_column_available = True


    # Drop universally irrelevant columns early
    df_corr.drop(columns=['id', 'Thank you for filling out our form! <3', 'Timestamp'], inplace=True, errors='ignore')

    # Filter by relevant professions (keeping this as it focuses the dataset)
    if profession_column_available:
        # Ensure 'profession' column exists before filtering
        if 'profession' in df_corr.columns:
            df_corr = df_corr[df_corr['profession'].isin(['Data Science', 'Software Development'])].copy(deep=True)
        else:
             # This case should ideally not happen if profession_column_available is True
             print("Error: 'profession' column not found despite check.")

    # --- Define the specific features to include in the correlation matrix ---
    # These are the original column names before any encoding/transformation
    # Curated list - removed some potentially less central OHE features to reduce size
    features_to_include_original = [
        # Demographics & Experience
        'What is your age?',
        'What is your gender?',
        'How many years of programming experience do you have?',
        'How often do you switch between these languages?',
        # 'How often do you seek help from online resources (e.g., Stack Overflow, official documentation)?', # Removed to shorten

        # Language Usage & Preference
        'Which language do you use most frequently?', # Will be OHE
        '  Which language do you prefer for the following tasks?   [Data Science & Machine Learning]', # Will be OHE
        '  Which language do you prefer for the following tasks?   [Web Development]', # Will be OHE
        '  Which language do you prefer for the following tasks?   [Mobile App Development]', # Will be OHE
        '  Which language do you prefer for the following tasks?   [Enterprise Applications]', # Will be OHE
        '  Which language do you prefer for the following tasks?   [Statistical Analysis]', # Will be OHE
        'Which language do you perceive as the most efficient for your tasks?', # Will be OHE
        'Which language was the easiest for you to learn?', # Will be OHE
        'Which language has the most beginner-friendly documentation and learning resources?', # Will be OHE

        # Job Market / Industry Perception
        'In your field, which language is the most commonly used?', # Will be OHE
        'Which language do you believe is most in demand for jobs?', # Will be OHE
        'Which language would you recommend for someone entering your industry?', # Will be OHE

        # Ratings
        'Python_Community_Support',
        'Python_Execution_Speed',
        'Python_Ease_of_Use',
        'Python_Documentation',
        'Python_Concurrency_Features', # Might remove Concurrency if needed for space
        'Python_Readability',
        'Java_Community_Support',
        'Java_Execution_Speed',
        'Java_Ease_of_Use',
        'Java_Documentation',
        'Java_Concurrency_Features', # Might remove Concurrency if needed for space
        'Java_Readability',
        # Keep R ratings? May add clutter if correlations aren't strong. Let's keep for now.
        'R_Community_Support',
        'R_Execution_Speed',
        'R_Ease_of_Use',
        'R_Documentation',
        'R_Concurrency_Features',
        'R_Readability',


        # Open Source Contribution (Keep Yes/No/Planning if useful)
        'Have you ever contributed to an open-source project in any of these languages?', # Will be OHE (Yes/No/Planning to/Unknown)
         # Languages_Used - This is a multi-select OHE, can add many columns. Let's remove for a cleaner matrix.
         # 'Languages_Used', # Removed
    ]

    # Include profession if it exists and the filter was applied successfully
    if profession_column_available and 'profession' in df_corr.columns:
         # Add it at the start or end of the list as needed
         features_to_include_original.insert(0, 'profession') # Add profession at the beginning


    # Filter df_corr to only keep these columns
    # Use errors='ignore' in case some columns from the list don't exist in the data after filtering
    # Ensure columns actually exist before selecting
    features_to_include_actual = [col for col in features_to_include_original if col in df_corr.columns]
    df_corr_subset = df_corr[features_to_include_actual].copy(deep=True)


    # --- Apply Transformations (Label Encoding and One-Hot Encoding) only to the subset ---

    # Define which of the *included* features need label encoding (numeric/ordinal)
    # This list should match the columns in features_to_include_actual that require label encoding
    label_encode_features_subset = [
         col for col in [
                         'What is your age?',
                         'How many years of programming experience do you have?',
                         'How often do you switch between these languages?',
                         'How often do you seek help from online resources (e.g., Stack Overflow, official documentation)?', # Re-added here if kept above
                         'Python_Community_Support',
                         'Python_Execution_Speed',
                         'Python_Ease_of_Use',
                         'Python_Documentation',
                         'Python_Concurrency_Features',
                         'Python_Readability',
                         'Java_Community_Support',
                         'Java_Execution_Speed',
                         'Java_Ease_of_Use',
                         'Java_Documentation',
                         'Java_Concurrency_Features',
                         'Java_Readability',
                         'R_Community_Support',
                         'R_Execution_Speed',
                         'R_Ease_of_Use',
                         'R_Documentation',
                         'R_Concurrency_Features',
                         'R_Readability'] if col in df_corr_subset.columns
    ]

    for feature in label_encode_features_subset:
         # Apply label encoding only if the column is in the subset
         if feature in df_corr_subset.columns:
             df_corr_subset[feature] = label_encode_with_exceptions(df_corr_subset[feature])
         else:
              print(f"Warning: Label encode feature '{feature}' not found in subset.")


    # Define which of the *included* features need one-hot encoding (categorical)
    # This list should match the columns in features_to_include_actual that require OHE
    ohe_features_subset = [
        col for col in [
            'What is your gender?',
            # 'Languages_Used', # Removed
            '  Which language do you prefer for the following tasks?   [Data Science & Machine Learning]',
            '  Which language do you prefer for the following tasks?   [Web Development]',
            '  Which language do you prefer for the following tasks?   [Mobile App Development]',
            '  Which language do you prefer for the following tasks?   [Enterprise Applications]',
            '  Which language do you prefer for the following tasks?   [Statistical Analysis]',
            'Which language do you use most frequently?',
            'Which language do you perceive as the most efficient for your tasks?',
            'Which language do you think will dominate the job market in the next 5 years?',
            'Which language was the easiest for you to learn?',
            'Which language has the most beginner-friendly documentation and learning resources?',
            'In your field, which language is the most commonly used?',
            'Which language do you believe is most in demand for jobs?',
            'Which language would you recommend for someone entering your industry?',
            'Have you ever contributed to an open-source project in any of these languages?',
            'profession' # Add profession here if it's included and categorical
        ] if col in df_corr_subset.columns
    ]

    one_hot_encoder = OneHotEncoder(sparse_output=False, drop='if_binary', handle_unknown='ignore')

    # Start with label-encoded columns, then add OHE columns
    # Need to make sure df_processed_subset is initialized correctly even if label_encode_features_subset is empty
    if label_encode_features_subset:
         df_processed_subset = df_corr_subset[label_encode_features_subset].copy()
    else:
         # Create an empty DataFrame with the same index if no label-encoded features
         df_processed_subset = pd.DataFrame(index=df_corr_subset.index)


    for column in ohe_features_subset:
         # Apply OHE only if the column is in the subset
         if column in df_corr_subset.columns:
             # Ensure column data type is string before one-hot encoding
             df_corr_subset[column] = df_corr_subset[column].astype(str)
             encoded_features = one_hot_encoder.fit_transform(df_corr_subset[[column]])
             encoded_feature_names = one_hot_encoder.get_feature_names_out([column])
             encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df_corr_subset.index)

             # Concatenate OHE columns to the processed subset DataFrame
             df_processed_subset = pd.concat([df_processed_subset, encoded_df], axis=1)
         else:
             print(f"Warning: OHE feature '{column}' not found in subset.")


    # Drop columns with 'Not Applicable' or 'nan' in their name that resulted from OHE
    columns_to_drop_encoded = [col for col in df_processed_subset.columns if 'Not Applicable' in col or 'nan' in col.lower()]
    df_processed_subset.drop(columns=columns_to_drop_encoded, inplace=True, errors='ignore')


    # --- Prepare for Correlation Calculation ---
    # Ensure all columns are numeric. Coerce errors (like original NaNs not caught by label_encode) to NaN
    for col in df_processed_subset.columns:
        # Replace the NaN placeholder from label encoding with NaN if that was the choice
        # The label_encode_with_exceptions function should already handle mapping exceptions to NaN (pd.NA/np.nan)
        # So just ensure the dtype is numeric
        df_processed_subset[col] = pd.to_numeric(df_processed_subset[col], errors='coerce')

    # Drop any columns that became all NaN after conversion (e.g., if an OHE column had no valid data)
    df_processed_subset.dropna(axis=1, how='all', inplace=True)
    df_processed_subset.dropna(axis=0, how='all', inplace=True) # Also drop rows that became all NaN


    # Check if there's data left to correlate
    if df_processed_subset.empty or df_processed_subset.shape[1] < 2:
        correlation_plot_div = "<p>Insufficient data or features after filtering and processing to compute correlation matrix.</p>"
    else:
        # Compute the correlation matrix on the processed subset
        correlation_matrix = df_processed_subset.corr()

        # --- Apply the threshold and filtering as before ---
        threshold = 0.5
        filtered_correlation_matrix = correlation_matrix.where(correlation_matrix.abs() >= threshold)

        # Drop rows and columns with all NaNs *after* applying the threshold
        filtered_correlation_matrix.dropna(how='all', axis=0, inplace=True)
        filtered_correlation_matrix.dropna(how='all', axis=1, inplace=True)

        # Check again if matrix is empty after filtering by threshold
        if filtered_correlation_matrix.empty:
             correlation_plot_div = "<p>No correlations above the specified threshold (0.5) found among the selected features.</p>"
        else:
            # Apply the shortening function to the index and column names of the filtered matrix
            filtered_correlation_matrix.index = filtered_correlation_matrix.index.map(shorten_feature_name)
            filtered_correlation_matrix.columns = filtered_correlation_matrix.columns.map(shorten_feature_name)

            # --- Determine appropriate heatmap size based on the number of features ---
            num_features = len(filtered_correlation_matrix.columns)
            # Adjusted sizing parameters for a more compact view
            base_size = 500  # Smaller base size
            size_per_feature = 25 # Reduced pixels per feature

            # Calculate dynamic size, with a minimum. Add buffer for labels/margins.
            heatmap_width = max(base_size, num_features * size_per_feature ) # Increased margin buffer slightly for potentially longer shortened names
            heatmap_height = max(base_size, num_features * size_per_feature )


            # Create the Plotly Heatmap
            fig_corr = go.Figure(data=go.Heatmap(
                    z=filtered_correlation_matrix.values,
                    x=filtered_correlation_matrix.columns.tolist(),
                    y=filtered_correlation_matrix.index.tolist(),
                    colorscale='RdBu',
                    zmin=-1,
                    zmax=1,
                    colorbar=dict(title='Correlation', titlefont=dict(color='white'), tickfont=dict(color='white')), # Style colorbar
                    hovertemplate='<b>%{x}</b><br><b>%{y}</b><br>Correlation: %{z:.2f}<extra></extra>'
                ))

            # Add annotations (correlation values) - Only add if the matrix is small enough
            # Further reduced the threshold for annotations to avoid overwhelming the plot
            if num_features > 0 and num_features < 20: # Adjusted threshold for annotations again
                annotations = []
                # Iterate through the filtered matrix values
                for i in range(filtered_correlation_matrix.shape[0]):
                    for j in range(filtered_correlation_matrix.shape[1]):
                         value = filtered_correlation_matrix.iloc[i, j]
                         col_name = filtered_correlation_matrix.columns[j]
                         row_name = filtered_correlation_matrix.index[i]
                         # Check if value is a number and not NaN before adding annotation
                         if pd.notna(value) and np.isscalar(value) and abs(value) >= threshold:
                             annotations.append(
                                 dict(x=col_name, y=row_name, text=f"{value:.2f}",
                                      # Adjust annotation font color based on background color (RdBu scale)
                                      # Dark text on light areas, white text on dark areas
                                      # Correlation near 0 is white/light gray, near -1 or 1 is dark red/blue
                                      font=dict(color="black" if abs(value) < 0.5 else "white", size=7), # Adjusted annotation font size
                                      showarrow=False)
                             )
                if annotations: # Only update layout if there are annotations to add
                    fig_corr.update_layout(annotations=annotations)
                else:
                     print("No annotations meeting criteria found.")
            else:
                print(f"Too many features ({num_features}) for annotations or matrix is empty. Skipping annotations.")


            # Update layout for styling and sizing
            fig_corr.update_layout(
                title="Filtered Correlation Matrix Heatmap (Abs Correlation >= 0.5)<br><i>(Selected Features)</i>",
                xaxis_title="Features",
                yaxis_title="Features",
                xaxis={'side': 'bottom'},
                # Adjust margins to make room for labels
                margin=dict(l=280, r=50, t=100, b=280), # Increased margins significantly for rotated shortened labels
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=9), # Adjusted base font size
                title_font=dict(color='white', size=14), # Adjust title font size
                width=heatmap_width,   # Set explicit width
                height=heatmap_height # Set explicit height
            )

            # Convert figure to HTML
            correlation_plot_div = to_html(fig_corr, full_html=False)


    # The rest of the return statement remains the same
    return templates.TemplateResponse("correlation.html", {
        "request": request,
        "correlation_plot_div": correlation_plot_div,
        'host': host,
        'port': port
    })












df_original = pd.read_csv('./cleaned_full_survey_data.csv')
profession_df = pd.read_csv("processed-datasets/profession_categories.csv")

# Ensure df_infer is a copy for the infer_stat route's modifications
df_infer = df_original.copy()
# Ensure profession_df is aligned with df_infer if needed for infer_stat

# Define the label encoding function here so it's accessible to ML route
def label_encode_with_exceptions(series, exception='Not Applicable'):
    mask = series != exception
    encoder = LabelEncoder()
    # Fit and transform only on the masked data
    # Handle potential errors if mask results in empty series for fit_transform
    if series[mask].empty:
         encoded = np.array([])
         original_masked_index = []
    else:
        original_masked_index = series[mask].index
        encoded = encoder.fit_transform(series[mask].astype(str)) # Ensure data is string type for consistent encoding

    # Create a result Series with original index and fill non-masked values
    result = pd.Series(index=series.index, dtype=float) # Use float to allow for NaN
    result.loc[original_masked_index] = encoded
    # Handle the 'Not Applicable' values and original NaNs
    result[~mask] = np.nan # Use NaN for 'Not Applicable' or original NaNs

    return result


# Define the function to get the most used language from OHE columns
def get_most_used_language(row):
    # Use errors='ignore' in get to handle cases where columns might be missing unexpectedly
    if row.get('Which language do you use most frequently?_Python', 0) == 1:
        return 'Python'
    elif row.get('Which language do you use most frequently?_Java', 0) == 1:
        return 'Java'
    elif row.get('Which language do you use most frequently?_All equally', 1) == 1: # Assume All equally is 1 if present
        return 'All equally'
    else:
        return 'Unknown'

# Define a function to shorten feature names for plots
def shorten_feature_name(name):
    name = name.replace('Which language do you prefer for the following tasks?   [', '')
    name = name.replace('Which language do you perceive as the most efficient for your tasks?', 'Perceived Most Efficient')
    name = name.replace('Which language do you think will dominate the job market in the next 5 years?', 'Future Dominance')
    name = name.replace('Which language was the easiest for you to learn?', 'Easiest to Learn')
    name = name.replace('Which language has the most beginner-friendly documentation and learning resources?', 'Best Docs & Resources')
    name = name.replace('In your field, which language is the most commonly used?', 'Most Used in Field')
    name = name.replace('Which language do you believe is most in demand for jobs?', 'Most In Demand Jobs')
    name = name.replace('Which language would you recommend for someone entering your industry?', 'Recommended for Entry')
    name = name.replace('Have you ever contributed to an open-source project in any of these languages?', 'OSS Contribution')
    name = name.replace('How often do you switch between these languages?', 'Switch Frequency')
    name = name.replace('How many years of programming experience do you have?', 'Years Experience')
    name = name.replace('What is your age?', 'Age')
    name = name.replace('How often do you seek help from online resources (e.g., Stack Overflow, official documentation)?', 'Seek Help Frequency')


    # Handle one-hot encoded suffixes last
    name = name.replace(']_Python', ' (Py)')
    name = name.replace(']_Java', ' (Java)')
    name = name.replace(']_R', ' (R)') # Add R suffix
    name = name.replace(']_nan', ' (NaN)') # Handle OHE nan columns
    name = name.replace('?_Male', ' (Male)')
    name = name.replace('?_Female', ' (Female)')
    name = name.replace('?_Non-binary', ' (Non-binary)')
    name = name.replace('?_Yes', ' (Yes)')
    name = name.replace('?_No', ' (No)')
    name = name.replace('profession_Data Science', 'Role (DS)')
    name = name.replace('profession_Software Development', 'Role (SWD)')
    name = name.replace('Languages_Used_', 'Used: ')
    name = name.replace('?_It depends', ' (Depends)')
    name = name.replace('?_All equally', ' (All Equal)') # Suffix for All equally
    name = name.replace('Python_Community_Support', 'Py Community Support')
    name = name.replace('Java_Community_Support', 'Java Community Support')
    name = name.replace('Python_Execution_Speed', 'Py Speed')
    name = name.replace('Java_Execution_Speed', 'Java Speed')
    name = name.replace('R_Execution_Speed', 'R Speed')
    name = name.replace('Python_Ease_of_Use', 'Py Ease of Use')
    name = name.replace('Java_Ease_of_Use', 'Java Ease of Use')
    name = name.replace('R_Ease_of_Use', 'R Ease of Use')
    name = name.replace('Python_Documentation', 'Py Docs')
    name = name.replace('Java_Documentation', 'Java Docs')
    name = name.replace('Python_Concurrency_Features', 'Py Concurrency')
    name = name.replace('Java_Concurrency_Features', 'Java Concurrency')
    name = name.replace('Python_Readability', 'Py Readability')
    name = name.replace('Java_Readability', 'Java Readability')


    # Clean up any remaining brackets or extra spaces
    name = name.replace('[', '').replace(']', '').strip()


    return name


# Define the machine learning route
@app.get('/machine_learning', response_class=HTMLResponse)
async def run_machine_learning(request: Request):
    # Make a copy of the original DataFrame for ML processing
    df_ml = df_original.copy()

    # --- Data Preprocessing (Replicate from Notebook) ---
    df_ml['profession'] = profession_df['Predicted_Category']
    df_ml.drop(columns=['id', 'What is your current major or job field?', 'Thank you for filling out our form! <3', 'Timestamp'], inplace=True, errors='ignore')
    df_ml = df_ml[df_ml['profession'].isin(['Data Science', 'Software Development'])].copy(deep=True)
    df_ml['Which language would you recommend for someone entering your industry?'] = df_ml['Which language would you recommend for someone entering your industry?'].astype(str)

    features_to_transform = ['Python_Community_Support',
                         'How often do you switch between these languages?',
                         'How many years of programming experience do you have?',
                         'What is your age?',
                        'Java_Community_Support',
                        'Java_Execution_Speed',
                        'R_Execution_Speed',
                        'Python_Execution_Speed',
                        'Python_Ease_of_Use',
                        'R_Ease_of_Use',
                        'Java_Ease_of_Use',
                        'Python_Documentation',
                        'Java_Documentation',
                        'Python_Concurrency_Features',
                        'Java_Concurrency_Features',
                        'Python_Readability',
                        'Java_Readability',
                        'How often do you seek help from online resources (e.g., Stack Overflow, official documentation)?'
                        ]

    for feature in features_to_transform:
        if feature in df_ml.columns:
            df_ml[feature] = label_encode_with_exceptions(df_ml[feature])
        else:
            print(f"Warning: Feature '{feature}' not found for label encoding in ML route.")

    one_hot_encoder = OneHotEncoder(sparse_output=False, drop='if_binary', handle_unknown='ignore')
    columns_to_encode = [
        'What is your gender?',
        'profession',
        'Which language would you recommend for someone entering your industry?',
        'Languages_Used',
        '  Which language do you prefer for the following tasks?   [Data Science & Machine Learning]',
        'Which language do you use most frequently?',
        'Which language do you perceive as the most efficient for your tasks?',
        '  Which language do you prefer for the following tasks?   [Web Development]',
        '  Which language do you prefer for the following tasks?   [Mobile App Development]',
        '  Which language do you prefer for the following tasks?   [Enterprise Applications]',
        '  Which language do you prefer for the following tasks?   [Statistical Analysis]',
        'Which language do you think will dominate the job market in the next 5 years?',
        'Which language was the easiest for you to learn?',
        'Which language has the most beginner-friendly documentation and learning resources?',
        'In your field, which language is the most commonly used?',
        'Which language do you believe is most in demand for jobs?',
        'Have you ever contributed to an open-source project in any of these languages?'
    ]

    for column in columns_to_encode:
        if column in df_ml.columns:
            df_ml[column] = df_ml[column].astype(str)
            encoded_features = one_hot_encoder.fit_transform(df_ml[[column]])
            # Ensure column names are unique if multiple OHE columns are generated
            encoded_feature_names = one_hot_encoder.get_feature_names_out([column])
            encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df_ml.index)

            # Handle potential duplicate column names before concat
            # This can happen if original df_ml already had columns with the same name as the OHE output
            # (though less likely with survey data like this)
            # A safer way is to ensure original columns that would be encoded are dropped first
            # which is already done by the `df_ml.drop(columns=[column], inplace=True)` below.
            # But if get_feature_names_out produces names that clash with *other* existing columns...
            # For now, let's rely on the drop below and assume no tricky name clashes.

            df_ml = pd.concat([df_ml, encoded_df], axis=1)
            df_ml.drop(columns=[column], inplace=True)
        else:
            print(f"Warning: Column '{column}' not found for one-hot encoding in ML route.")


    # Drop columns with 'Not Applicable' or 'nan' in their name (resulting from one-hot encoding)
    # Also drop original columns that were label encoded as they are now numerical
    columns_to_drop_encoded = [col for col in df_ml.columns if 'Not Applicable' in col or 'nan' in col.lower()]
    df_ml.drop(columns=columns_to_drop_encoded, inplace=True, errors='ignore')

    columns_to_drop_additional = [
        'Have you ever had to switch from one of these languages to another due to project requirements? Why?',
        'If you could only use one of these three languages for all future projects, which would you choose and why?',
        'What do you think is the biggest advantage of your preferred language?',
        'What is your current role?',
        'Python_Optimization_Challenges',
        'Python_Learning_Challenges',
        'Java_Optimization_Challenges',
        'Java_Learning_Challenges',
        'Would you recommend learning another language? Why?',
         # These columns are now expected to be in the DataFrame AFTER label encoding,
         # so we should NOT drop them here unless explicitly intended for ML features.
         # 'Python_Community_Support', # These might be numeric (-1 or encoded) - handled by label encoding
         # 'How often do you switch between these languages?',
         # 'How many years of programming experience do you have?',
         # 'What is your age?',
         # 'Java_Community_Support',
         # 'Java_Execution_Speed',
         # 'R_Execution_Speed',
         # 'Python_Execution_Speed',
         # 'Python_Ease_of_Use',
         # 'R_Ease_of_Use',
         # 'Java_Ease_of_Use',
         # 'Python_Documentation',
         # 'Java_Documentation',
         # 'Python_Concurrency_Features',
         # 'Java_Concurrency_Features',
         # 'Python_Readability',
         # 'Java_Readability',
         # 'How often do you seek help from online resources (e.g., Stack Overflow, official documentation)?'

    ]
     # Filter columns_to_drop_additional to exclude columns that might have been label encoded and are needed as features
    # Assuming features_to_transform contains the columns that were label encoded and should *not* be dropped here
    columns_to_drop_additional_filtered = [col for col in columns_to_drop_additional if col in df_ml.columns and col not in features_to_transform]
    df_ml.drop(columns=columns_to_drop_additional_filtered, inplace=True, errors='ignore')


    # --- Prepare Data for ML ---
    # Replace any remaining pd.NA (from label encoding NaNs or 'Not Applicable') with 0 or a suitable strategy
    # Note: Replacing with 0 after label encoding might not be ideal if -1 was used for 'Not Applicable'
    # Let's refine label_encode_with_exceptions to return NaN, then fill NaNs here.
    # Replicating notebook's fillna(0) approach for now, but acknowledge it might not be optimal ML practice.
    # df_ml = df_ml.replace(pd.NA, 0).replace(-1, 0) # Assumes -1 should also become 0
    # A safer fill after ensuring NaNs from label_encode_with_exceptions are proper pd.NA
    df_ml = df_ml.fillna(0) # Fill all remaining NaNs (including those from label encoding exceptions) with 0


    # Create the target variable based on one-hot encoded columns
    # Need to make sure these columns exist AFTER one-hot encoding and before dropping
    target_columns_ohe_check = [
        'Which language do you use most frequently?_Python',
        'Which language do you use most frequently?_Java',
        'Which language do you use most frequently?_All equally'
    ]
    # Ensure target columns exist before creating target_language
    if all(col in df_ml.columns for col in target_columns_ohe_check):
        df_ml['target_language'] = df_ml.apply(get_most_used_language, axis=1)
        df_ml = df_ml[df_ml['target_language'] != 'Unknown'].copy(deep=True) # Ensure df_ml is a copy after filtering
        target_available = True
    else:
        print("Error: Required OHE target columns not found for target creation.")
        target_available = False
        df_ml = pd.DataFrame() # Clear dataframe if target cannot be created


    # Separate features (X) and target (y)
    if target_available and not df_ml.empty:
        X = df_ml.drop(columns=target_columns_ohe_check + ['target_language'], errors='ignore')
        y = df_ml['target_language'].map({'Python': 0, 'Java': 1, 'All equally': 2})

        # Drop any non-numeric columns that might have slipped through
        X = X.select_dtypes(include=['number'])

        # Align X and y indices if any rows were dropped
        y = y.loc[X.index]

        # Ensure no NaNs or infinity in X and y before ML steps
        # Drop rows where target is NaN (shouldn't happen if 'Unknown' is filtered)
        valid_indices = y.dropna().index
        X = X.loc[valid_indices].dropna() # Drop rows with NaNs in features
        y = y.loc[X.index] # Align y again after dropping rows from X
        X = X.replace([float('inf'), float('-inf')], 0) # Replace inf values
        y = y.replace([float('inf'), float('-inf')], 0) # Replace inf values in y if any

        print(f"Shape after cleaning NaNs/Infs: X={X.shape}, y={y.shape}")

    else:
        print("Cannot prepare data for ML: Target not available or DataFrame is empty.")
        X = pd.DataFrame()
        y = pd.Series()


    # --- Feature Selection using SelectKBest ---
    feature_scores_plot_div = "<p>Insufficient data for feature selection.</p>" # Default placeholder
    selected_features_names = [] # To store names of selected features
    X_ml_processed = X.copy() # Start with cleaned X, might be updated by selection
    y_ml_processed = y.copy() # Start with cleaned y, might be updated by selection


    if not X.empty and len(X.columns) > 0 and not y.empty and len(y.unique()) > 1: # Need at least 2 classes for chi2
        try:
            # Adjust k based on the number of available features
            k_features = min(20, X.shape[1])
            if k_features > 0:
                bestfeatures = SelectKBest(score_func=chi2, k=k_features)
                fit = bestfeatures.fit(X, y) # Use cleaned X, y

                dfscores = pd.DataFrame(fit.scores_)
                dfcolumns = pd.DataFrame(X.columns)

                featureScores = pd.concat([dfcolumns, dfscores], axis=1)
                featureScores.columns = ['Feature', 'Score']
                featureScores = featureScores.sort_values(by='Score', ascending=False).head(k_features) # Sort and take top k

                # Apply the shortening function to feature names for the plot
                featureScores['Feature'] = featureScores['Feature'].apply(shorten_feature_name)


                # Plotly Bar Chart for Feature Scores
                fig_feature_scores = px.bar(featureScores,
                                            x='Score',
                                            y='Feature',
                                            orientation='h',
                                            title=f'Top {k_features} Feature Scores (Chi-squared)',
                                            labels={'Score': 'Chi-squared Score', 'Feature': 'Feature Name'})
                fig_feature_scores.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    title_font=dict(color='white'),
                    xaxis=dict(tickfont=dict(color='white')),
                    yaxis=dict(tickfont=dict(color='white')),
                    margin=dict(l=250, r=50, t=50, b=50), # Add left margin for long labels
                    height= max(400, k_features * 30) # Adjust height based on number of features
                )
                feature_scores_plot_div = to_html(fig_feature_scores, full_html=False)

                # Transform the data to selected features
                selected_indices = fit.get_support(indices=True)
                X_ml_processed = fit.transform(X) # Use the selected features for ML
                selected_features_names = X.columns[selected_indices].tolist() # Store original names of selected features
                y_ml_processed = y.copy() # y remains the same after selection


            else: # k_features is 0
                feature_scores_plot_div = "<p>Not enough features (k=0) for selection after cleaning.</p>"
                X_ml_processed = X.copy() # Use all cleaned features
                y_ml_processed = y.copy()
                selected_features_names = X.columns.tolist() # All features considered 'selected'
                k_features = len(selected_features_names) # Update k_features

        except ValueError as e:
            print(f"Error during feature selection (chi2 requires non-negative features): {e}")
            feature_scores_plot_div = f"<p>Error during feature selection (chi2 requires non-negative features): {e}</p>"
            X_ml_processed = X.copy() # Use cleaned data without selection if error occurs
            y_ml_processed = y.copy()
            selected_features_names = X.columns.tolist() # No features selected, use all
            k_features = len(selected_features_names)

        except Exception as e:
            print(f"An unexpected error occurred during feature selection: {e}")
            feature_scores_plot_div = f"<p>An unexpected error occurred during feature selection: {e}</p>"
            X_ml_processed = X.copy() # Use cleaned data without selection if error occurs
            y_ml_processed = y.copy()
            selected_features_names = X.columns.tolist() # No features selected, use all
            k_features = len(selected_features_names)

    else:
         print("Skipping feature selection: X is empty, has no columns, y is empty, or y has only one class.")
         # Use cleaned X and y directly for subsequent steps
         X_ml_processed = X.copy()
         y_ml_processed = y.copy()
         selected_features_names = X.columns.tolist() # All features used
         k_features = len(selected_features_names)


    # --- Resampling using TomekLinks ---
    # TomekLinks removes pairs of instances from different classes that are very close
    # It's generally used for *under*-sampling the majority class
    # Ensure we have enough data and multiple classes before resampling
    X_resampled, y_resampled = X_ml_processed, y_ml_processed # Start with processed data

    if X_ml_processed.shape[0] > 10 and len(y_ml_processed.unique()) > 1: # Needs sufficient data and > 1 class
        try:
            oversample = TomekLinks() # This might reduce the majority class sample size
            print(f"Original dataset shape before TomekLinks: {X_ml_processed.shape}, {y_ml_processed.shape}")
            X_resampled, y_resampled = oversample.fit_resample(X_ml_processed, y_ml_processed)
            print(f"Resampled dataset shape after TomekLinks: {X_resampled.shape}, {y_resampled.shape}")
        except Exception as e:
            print(f"Error during resampling (TomekLinks): {e}")
            # If resampling fails, continue with the data before resampling
            X_resampled, y_resampled = X_ml_processed, y_ml_processed
    else:
        print("Skipping resampling: Insufficient data or only one class.")


    # --- Train-Test Split ---
    model_accuracies_plot_div = "<p>Insufficient data for model training.</p>" # Default placeholder
    classification_reports_html = "<h3>Classification Reports</h3><p>Insufficient data for model training.</p>" # Default placeholder
    summary_report_html = "<p>Insufficient data for model training.</p>" # Default placeholder


    if X_resampled.shape[0] > 10 and len(y_resampled.unique()) > 1: # Need enough data and > 1 class for split
         # Ensure y_resampled has at least 2 unique values for stratification
         stratify_y = y_resampled if len(y_resampled.unique()) > 1 else None
         X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=stratify_y)

         print(f"Train shapes: X_train={X_train.shape}, y_train={y_train.shape}")
         print(f"Test shapes: X_test={X_test.shape}, y_test={y_test.shape}")


         # --- Model Training and Evaluation ---
         # Define models and parameter grids again within the scope where training happens
         models = {
             'Logistic Regression': LogisticRegression(max_iter=2000), # Increased max_iter
             'KNN': KNeighborsClassifier(),
             'Naive Bayes': GaussianNB(),
             'SVM': SVC(probability=True),
             'Decision Tree': DecisionTreeClassifier()
         }

         ensemble_models = {
             'Random Forest': RandomForestClassifier(random_state=42), # Added random_state
             'Extra Trees': ExtraTreesClassifier(random_state=42), # Added random_state
             'Bagging': BaggingClassifier(random_state=42), # Added random_state
             'AdaBoost': AdaBoostClassifier(random_state=42), # Added random_state
             'Gradient Boosting': GradientBoostingClassifier(random_state=42) # Added random_state
         }

         param_grids = {
             'Logistic Regression': {'C': [0.1, 1, 10]},
             'KNN': {'n_neighbors': [3, 5, 7]},
             'Naive Bayes': {},
             'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
             'Decision Tree': {'max_depth': [None, 5, 10]},
             'Random Forest': {'n_estimators': [50, 100, 200]},
             'Extra Trees': {'n_estimators': [50, 100, 200]},
             'Bagging': {'n_estimators': [10, 20, 30]},
             'AdaBoost': {'n_estimators': [50, 100, 200]},
             'Gradient Boosting': {'n_estimators': [50, 100, 200]}
         }

         cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

         best_models = {}
         evaluation_results = [] # List to store evaluation summary and reports

         # Train and evaluate models
         all_model_names = list(models.keys()) + list(ensemble_models.keys())
         for name in all_model_names:
             try:
                 if name in models:
                     model = models[name]
                     param_grid = param_grids[name]
                 else: # Ensemble models
                     model = ensemble_models[name]
                     param_grid = param_grids[name]

                 # Skip GridSearchCV if param_grid is empty (like for Naive Bayes)
                 if param_grid:
                      grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy')
                      grid_search.fit(X_train, y_train)
                      best_models[name] = grid_search.best_estimator_
                      best_params = grid_search.best_params_
                      train_cv_score = grid_search.best_score_
                 else: # No hyperparameters to tune, just train the model directly
                     model.fit(X_train, y_train)
                     best_models[name] = model
                     best_params = 'N/A (No grid search)'
                     # Calculate cross-val score manually if needed, or mark as N/A
                     # cross_val_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
                     # train_cv_score = cross_val_scores.mean()
                     train_cv_score = 'N/A' # For simplicity, mark as N/A if no grid search

                 y_pred = best_models[name].predict(X_test)
                 accuracy = accuracy_score(y_test, y_pred)
                 report_dict = classification_report(y_test, y_pred, output_dict=True)

                 evaluation_results.append({
                     'Model': name,
                     'Best Params': best_params,
                     'Train Cross-Val Score': train_cv_score,
                     'Test Accuracy': accuracy,
                     'Classification Report': report_dict
                 })
                 print(f"{name}: Best Parameters - {best_params}, Test Accuracy - {accuracy:.4f}")

             except Exception as e:
                 print(f"Error training or evaluating {name}: {e}")
                 evaluation_results.append({'Model': name, 'Error': str(e)})

         # Voting Classifier (assuming at least two required models were successfully trained)
         required_estimators = ['Random Forest', 'SVM'] # Models needed for Voting
         if all(model_name in best_models for model_name in required_estimators):
              try:
                  estimators = [(name, best_models[name]) for name in required_estimators] # Use trained best estimators
                  # Check if base estimators support probability prediction for soft voting
                  if all(hasattr(est, 'predict_proba') for name, est in estimators):
                       voting_clf = VotingClassifier(estimators=estimators, voting='soft')
                       voting_clf.fit(X_train, y_train)
                       best_models['Voting'] = voting_clf

                       y_pred = best_models['Voting'].predict(X_test)
                       accuracy = accuracy_score(y_test, y_pred)
                       report_dict = classification_report(y_test, y_pred, output_dict=True)

                       evaluation_results.append({
                           'Model': 'Voting',
                           'Best Params': 'N/A (Voting)',
                           'Train Cross-Val Score': 'N/A',
                           'Test Accuracy': accuracy,
                           'Classification Report': report_dict
                       })
                       print(f"Voting: Test Accuracy - {accuracy:.4f}")
                  else:
                       print("Skipping Voting Classifier: Base estimators do not support predict_proba for soft voting.")
                       evaluation_results.append({'Model': 'Voting', 'Note': 'Skipped (Base estimators lack predict_proba for soft voting).'})
              except Exception as e:
                  print(f"Error training or evaluating Voting Classifier: {e}")
                  evaluation_results.append({'Model': 'Voting', 'Error': str(e)})
         else:
             print("Skipping Voting Classifier: Required base models ('Random Forest', 'SVM') not trained successfully.")
             evaluation_results.append({'Model': 'Voting', 'Note': 'Skipped (Required base models not trained).'})


         # --- Plotly Visualization of Model Accuracies ---
         accuracy_data = [{'Model': res['Model'], 'Test Accuracy': res.get('Test Accuracy', 0)}
                            for res in evaluation_results if 'Test Accuracy' in res] # Use 0 if accuracy is missing
         accuracy_df = pd.DataFrame(accuracy_data)

         if not accuracy_df.empty:
             # Sort by accuracy for better visualization
             accuracy_df = accuracy_df.sort_values(by='Test Accuracy', ascending=False)

             fig_accuracies = px.bar(accuracy_df,
                                    x='Model',
                                    y='Test Accuracy',
                                    title='Model Test Set Accuracy',
                                    labels={'Test Accuracy': 'Accuracy'},
                                    color='Model') # Color bars by model
             fig_accuracies.update_layout(
                 plot_bgcolor='rgba(0,0,0,0)',
                 paper_bgcolor='rgba(0,0,0,0)',
                 font=dict(color='white'),
                 title_font=dict(color='white'),
                 xaxis=dict(tickangle=45, tickfont=dict(color='white')),
                 yaxis=dict(tickfont=dict(color='white'), range=[0, 1.05]), # Set y-axis range slightly above 1
                 margin=dict(l=50, r=50, t=50, b=100) # Adjust margins
             )
             model_accuracies_plot_div = to_html(fig_accuracies, full_html=False)
         else:
             model_accuracies_plot_div = "<p>No model accuracy results to display.</p>"


         # --- Prepare Classification Report Data for Display ---

         # 1. Summary Table
         summary_data = []
         for res in evaluation_results:
             if 'Classification Report' in res:
                 report = res['Classification Report']
                 # Extract key metrics, handling potential missing keys gracefully
                 summary_row = {
                     'Model': res['Model'],
                     'Test Accuracy': f"{res.get('Test Accuracy', np.nan):.4f}" if isinstance(res.get('Test Accuracy'), (int, float)) else res.get('Test Accuracy', 'N/A'),
                     'Macro Avg Precision': f"{report.get('macro avg', {}).get('precision', np.nan):.4f}" if isinstance(report.get('macro avg', {}).get('precision'), (int, float)) else 'N/A',
                     'Macro Avg Recall': f"{report.get('macro avg', {}).get('recall', np.nan):.4f}" if isinstance(report.get('macro avg', {}).get('recall'), (int, float)) else 'N/A',
                     'Macro Avg F1-Score': f"{report.get('macro avg', {}).get('f1-score', np.nan):.4f}" if isinstance(report.get('macro avg', {}).get('f1-score'), (int, float)) else 'N/A',
                     'Weighted Avg Precision': f"{report.get('weighted avg', {}).get('precision', np.nan):.4f}" if isinstance(report.get('weighted avg', {}).get('precision'), (int, float)) else 'N/A',
                     'Weighted Avg Recall': f"{report.get('weighted avg', {}).get('recall', np.nan):.4f}" if isinstance(report.get('weighted avg', {}).get('recall'), (int, float)) else 'N/A',
                     'Weighted Avg F1-Score': f"{report.get('weighted avg', {}).get('f1-score', np.nan):.4f}" if isinstance(report.get('weighted avg', {}).get('f1-score'), (int, float)) else 'N/A',
                     # Add Support column if needed, but it's usually the same for test set
                     # 'Support': report.get('macro avg', {}).get('support', 'N/A')
                 }
                 summary_data.append(summary_row)
             elif 'Error' in res:
                  summary_data.append({'Model': res['Model'], 'Error': res['Error']})
             elif 'Note' in res:
                  summary_data.append({'Model': res['Model'], 'Note': res['Note']})


         summary_df = pd.DataFrame(summary_data)

         if not summary_df.empty:
             # Convert summary DataFrame to HTML table
             summary_report_html = summary_df.to_html(index=False, classes='styled-table')
             # Add custom CSS class for styling in HTML
             summary_report_html = summary_report_html.replace('<table', '<table class="styled-table"')
         else:
              summary_report_html = "<p>No summary report data available.</p>"


         # 2. Detailed Reports (Formatted Text)
         detailed_reports_html = ""
         for res in evaluation_results:
             if 'Classification Report' in res:
                 report_dict = res['Classification Report']
                 model_name = res['Model']
                 # You can manually format the dictionary into a more readable text block
                 # or keep the JSON dump, just format it better
                 detailed_reports_html += f"<h4>{model_name} Detailed Report</h4>"
                 # Nicely format the JSON string for display
                 detailed_reports_html += "<pre class='detailed-report'>" + json.dumps(report_dict, indent=2) + "</pre>"
             elif 'Error' in res:
                  detailed_reports_html += f"<h4>{res['Model']} Detailed Report</h4><p>Error: {res['Error']}</p>"
             elif 'Note' in res:
                  detailed_reports_html += f"<h4>{res['Model']} Detailed Report</h4><p>Note: {res['Note']}</p>"


    else:
        print("Skipping model training and evaluation: Insufficient data after resampling.")
        model_accuracies_plot_div = "<p>Insufficient data for model training.</p>"
        classification_reports_html = "<h3>Classification Reports</h3><p>Insufficient data for machine learning.</p>"
        summary_report_html = "<p>Insufficient data for machine learning.</p>"


    return templates.TemplateResponse("machine_learning.html", {
        "request": request,
        "feature_scores_plot_div": feature_scores_plot_div,
        "model_accuracies_plot_div": model_accuracies_plot_div,
        "summary_report_html": summary_report_html,
        "detailed_reports_html": detailed_reports_html,
        'host': host,
        'port': port
    })


if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8000
    uvicorn.run("main:app", host=host, port=port, reload=True)