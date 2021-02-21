# Library Imports
from joblib import load
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import _pickle as pickle
from random import sample
from PIL import Image
from scipy.stats import halfnorm

# Loading the Profiles
with open("refined_profiles.pkl",'rb') as fp:
    df = pickle.load(fp)
    
with open("refined_cluster.pkl", 'rb') as fp:
    cluster_df = pickle.load(fp)
    
with open("vectorized_refined.pkl", 'rb') as fp:
    vect_df = pickle.load(fp)
    
# Loading the Classification Model
model = load("refined_model.joblib")

## Helper Functions

def string_convert(x):
    """
    First converts the lists in the DF into strings
    """
    if isinstance(x, list):
        return ' '.join(x)
    else:
        return x
 
    
def vectorization(df, columns, input_df):
    """
    Using recursion, iterate through the df until all the categories have been vectorized
    """

    column_name = columns[0]
        
    # Checking if the column name has been removed already
    if column_name not in ['Bios', 'Movies','Religion', 'Music', 'Politics', 'Social Media', 'Sports']:
                
        return df, input_df
    
    # Encoding columns with respective values
    if column_name in ['Religion', 'Politics']:
        
        # Getting labels for the original df
        df[column_name.lower()] = df[column_name].cat.codes
        
        # Dictionary for the codes
        d = dict(enumerate(df[column_name].cat.categories))
        
        d = {v: k for k, v in d.items()}
                
        # Getting labels for the input_df
        input_df[column_name.lower()] = d[input_df[column_name].iloc[0]]
                
        # Dropping the column names
        input_df = input_df.drop(column_name, 1)
        
        df = df.drop(column_name, 1)
        
        return vectorization(df, df.columns, input_df)
    
    # Vectorizing the other columns
    else:
        # Instantiating the Vectorizer
        vectorizer = CountVectorizer()
        
        # Fitting the vectorizer to the columns
        x = vectorizer.fit_transform(df[column_name].values.astype('U'))
        
        y = vectorizer.transform(input_df[column_name].values.astype('U'))

        # Creating a new DF that contains the vectorized words
        df_wrds = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names())
        
        y_wrds = pd.DataFrame(y.toarray(), columns=vectorizer.get_feature_names(), index=input_df.index)

        # Concating the words DF with the original DF
        new_df = pd.concat([df, df_wrds], axis=1)
        
        y_df = pd.concat([input_df, y_wrds], 1)

        # Dropping the column because it is no longer needed in place of vectorization
        new_df = new_df.drop(column_name, axis=1)
        
        y_df = y_df.drop(column_name, 1)
        
        return vectorization(new_df, new_df.columns, y_df) 

    
def scaling(df, input_df):
    """
    Scales the new data with the scaler fitted from the previous data
    """
    scaler = MinMaxScaler()
    
    scaler.fit(df)
    
    input_vect = pd.DataFrame(scaler.transform(input_df), index=input_df.index, columns=input_df.columns)
        
    return input_vect
    


def top_ten(cluster, vect_df, input_vect):
    """
    Returns the DataFrame containing the top 10 similar profiles to the new data
    """
    des_cluster = vect_df[vect_df['Cluster #']==cluster[0]].drop('Cluster #', 1)
    des_cluster = des_cluster.append(input_vect, sort=False)
    user_n = input_vect.index[0]
    corr = des_cluster.T.corrwith(des_cluster.loc[user_n])
    top_10_sim = corr.sort_values(ascending=False)[1:4]
    top_10 = df.loc[top_10_sim.index]
    top_10[top_10.columns[1:]] = top_10[top_10.columns[1:]]
    
    return top_10.astype('object')


def example_bios():
    """
    Creates a list of random example bios from the original dataset
    """
    st.write("-"*100)
    st.text("Some example Bios:\n(Try to follow the same format)")
    for i in sample(list(df.index), 3):
        st.text(df['Bios'].loc[i])
    st.write("-"*100)

# Creating a List for each Category

p = {}

movies = ['Adventure',
          'Action',
          'Drama',
          'Comedy',
          'Thriller',
          'Horror',
          'RomCom',
          'Musical',
          'Documentary']

p['Movies'] = [0.28,
               0.21,
               0.16,
               0.14,
               0.09,
               0.06,
               0.04,
               0.01, 
               0.01]

tv = ['Comedy',
      'Drama',
      'Action/Adventure',
      'Suspense/Thriller',
      'Documentaries',
      'Crime/Mystery',
      'News',
      'SciFi',
      'History']

p['TV'] = [0.30,
           0.23,
           0.12,
           0.12,
           0.09,
           0.08,
           0.03,
           0.02,
           0.01]

religion = ['Catholic',
            'Christian',
            'Jewish',
            'Muslim',
            'Hindu',
            'Buddhist',
            'Spiritual',
            'Other',
            'Agnostic',
            'Atheist']

p['Religion'] = [0.16,
                 0.16,
                 0.01,
                 0.19,
                 0.11,
                 0.05,
                 0.10,
                 0.09,
                 0.07,
                 0.06]

music = ['Rock',
         'HipHop',
         'Pop',
         'Country',
         'Latin',
         'EDM',
         'Gospel',
         'Jazz',
         'Classical']

p['Music'] = [0.30,
              0.23,
              0.20,
              0.10,
              0.06,
              0.04,
              0.03,
              0.02,
              0.02]


sports = ['Football',
          'Baseball',
          'Basketball',
          'Hockey',
          'Soccer',
          'Other']

p['Sports'] = [0.34,
               0.30,
               0.16, 
               0.13,
               0.04,
               0.03]


politics = ['Liberal',
            'Progressive',
            'Centrist',
            'Moderate',
            'Conservative']

p['Politics'] = [0.26,
                 0.11,
                 0.11,
                 0.15,
                 0.37]


social = ['Facebook',
          'Youtube',
          'Twitter',
          'Reddit',
          'Instagram',
          'Pinterest',
          'LinkedIn',
          'SnapChat',
          'TikTok']

p['Social Media'] = [0.36,
                     0.27,
                     0.11,
                     0.09,
                     0.05,
                     0.03,
                     0.03,
                     0.03,
                     0.03]

age = None

# Lists of Names and the list of the lists
categories = [movies, religion, music, politics, social, sports, age]

names = ['Movies','Religion', 'Music', 'Politics', 'Social Media', 'Sports', 'Age']

combined = dict(zip(names, categories))
    
    
## Interactive Section


st.title("Machine Learning Model for Dating App Demo for AppSynergy")

st.header("Finding a Partner with AI Using NaiveBayes, KNN and SVM")
st.write("Use Machine Learning to Find the Top 3 Dating Profile Matches")

image = Image.open('roshan_graffiti.png')
st.image(image, use_column_width=True)
new_profile = pd.DataFrame(columns=df.columns, index=[df.index[-1]+1])
new_profile['Bios'] = st.text_input("Tell us about yourself: ")       
example_bios()
random_vals = st.checkbox("Check here if you would like random values for yourself instead")
if random_vals:

    for i in new_profile.columns[1:]:
        if i in ['Religion', 'Politics']:  
            new_profile[i] = np.random.choice(combined[i], 1, p=p[i])
            
        elif i == 'Age':
            new_profile[i] = halfnorm.rvs(loc=18,scale=8, size=1).astype(int)
            
        else:
            new_profile[i] = list(np.random.choice(combined[i], size=(1,3), p=p[i]))
            
            new_profile[i] = new_profile[i].apply(lambda x: list(set(x.tolist())))

else:
    for i in new_profile.columns[1:]:
        if i in ['Religion', 'Politics']:  
            new_profile[i] = st.selectbox(f"Enter your choice for {i}", combined[i])
            
        elif i == 'Age':
            new_profile[i] = st.slider("What is your age?", 18, 50)
            
        else:
            options = st.multiselect(f"What is your preferred choice for {i}? (Pick up to 3)", combined[i])
            

            new_profile.at[new_profile.index[0], i] = options
            
            new_profile[i] = new_profile[i].apply(lambda x: list(set(x)))
            
            
for col in df.columns:
    df[col] = df[col].apply(string_convert)
    
    new_profile[col] = new_profile[col].apply(string_convert)
            


st.write("-"*1000)
st.write("Your profile:")
st.table(new_profile)

button = st.button("Click to find your Top 3!")

if button:    
    with st.spinner('Finding your Top 3 Matches...'):

        df_v, input_df = vectorization(df, df.columns, new_profile)
        new_df = scaling(df_v, input_df)
        cluster = model.predict(new_df)
        top_10_df = top_ten(cluster, vect_df, new_df)  
        st.success("Found your Top 3 Most Similar Profiles!")    
        st.balloons()
        st.table(top_10_df)
        

        

    

