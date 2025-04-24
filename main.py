from imports import FastAPI, HTTPException, HTMLResponse, px, to_html, pd, sns, Jinja2Templates, Request
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64
import io
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory="templates")


host = "127.0.0.1"
port = 8000



df = pd.read_csv('./cleaned_full_survey_data.csv')

@app.get('/')
def root():
    return {'test': 'done'}

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
    yaxis=dict(tickfont=dict(color='white'))

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
        'host':host,
        'port':port

    })


if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8000
    uvicorn.run("main:app", host=host, port=port, reload=True)