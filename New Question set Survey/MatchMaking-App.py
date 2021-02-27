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
with open("Survey_question.pkl",'rb') as fp:
    df = pickle.load(fp)
    
with open("Cluster.pkl", 'rb') as fp:
    cluster_df = pickle.load(fp)
    
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
    if column_name not in ['Idiology',
                         'Ethics,', 
                         'When it comes to Relationships', 
                         'Predictability always brings comfort', 
                         'Which is more important?', 
                         'When trust is broken?', 
                         'My close friend and family would say 1',
                         'My close friend and family would say 2',
                         'My close friend and family would say 3',
                         'Having common interests',
                         'Age']:
                
        return df, input_df
    
    # Encoding columns with respective values
    if column_name in ['Idiology']:
        df[column_name.lower()] = df[column_name].cat.codes
        d = dict(enumerate(df[column_name].cat.categories))
        d = {v: k for k, v in d.items()}
        input_df[column_name.lower()] = d[input_df[column_name].iloc[0]]
        input_df = input_df.drop(column_name, 1)
        df = df.drop(column_name, 1)
        return vectorization(df, df.columns, input_df)
    
    # Vectorizing the other columns
    else:
        vectorizer = CountVectorizer() 
        x = vectorizer.fit_transform(df[column_name].values.astype('U'))
        y = vectorizer.transform(input_df[column_name].values.astype('U'))
        df_wrds = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names())
        y_wrds = pd.DataFrame(y.toarray(), columns=vectorizer.get_feature_names(), index=input_df.index)
        new_df = pd.concat([df, df_wrds], axis=1)        
        y_df = pd.concat([input_df, y_wrds], 1)
        new_df = new_df.drop(column_name, axis=1)
        
        y_df = y_df.drop(column_name, 1)
        
        return vectorization(new_df, new_df.columns, y_df) 

    
def scaling(df, input_df):

    scaler = MinMaxScaler()
    
    scaler.fit(df)
    
    input_vect = pd.DataFrame(scaler.transform(input_df), index=input_df.index, columns=input_df.columns)
        
    return input_vect
    


def top(cluster, input_vect):
    des_cluster = des_cluster.append(input_vect, sort=False)
    user_n = input_vect.index[0]
    corr = des_cluster.T.corrwith(des_cluster.loc[user_n])
    top_10_sim = corr.sort_values(ascending=False)[1]
    top_10 = df.loc[top_10_sim.index]
    top_10[top_10.columns[1]] = top_10[top_10.columns[1]]
    
    return top_10.astype('object')

# Creating a List for each Category

p = {}

Idiology = ['opposites attracts','Similarity makes it smooth']

p['When it comes to love'] = [0, 1]

Ethics = ['If its right its easy', 'work is required']

p['When comes to love'] = [0, 1]

Relationship = ['Complicated','Communication is the Key']

p['When it comes to Relationships'] = [0, 1]

Predictability = ['yes', 'no']

p['Predictability always brings comfort'] = [0, 1]


Important = ['Work life balence',
             'Success']

p['Which is more important?'] = [0, 1]


Trust = ['The relationship is over',
         'There might be a chance']

p['When trust is broken?'] = [0, 1]


EDescription = ['Social butterfly',
                'Shy and reserved']

p['My close friend and family would say'] = [0, 1]

IDescription = ['Open book',
                'it takes a while']

p['My close friend and family would say'] = [0, 1]

Description = ['Strong Willed',
               'Laid back']

p['My close friend and family would say'] = [0, 1]

Interest = ['make or break a relation',
            'Doesnt match much']

p['Having common interests'] = [0, 1]

age = None

# Lists of Names and the list of the lists
categories = [Idiology, Ethics, Relationship, Predictability, Important, Trust, EDescription, IDescription, Description, Interest, age]

names = ['Idiology',
         'Ethics,', 
         'When it comes to Relationships', 
         'Predictability always brings comfort', 
         'Which is more important?', 
         'When trust is broken?', 
         'My close friend and family would say 1',
         'My close friend and family would say 2',
         'My close friend and family would say 3',
         'Having common interests',
         'Age']

combined = dict(zip(names, categories))
    
    
## Interactive Section


st.title("Machine Learning Model for Dating App Demo for AppSynergy")

st.header("Finding a Partner with AI Using NaiveBayes, KNN and SVM")
st.write("Use Machine Learning to Find the Top Dating Profile Matche")
# st.write(combined)
image = Image.open('roshan_graffiti.png')
st.image(image, use_column_width=True)
new_profile = pd.DataFrame(columns=df.columns, index=[df.index[-1]+1])
random_vals = st.checkbox("Check here if you would like random values for yourself instead")
if random_vals:

    for i in new_profile.columns[1:]:
        if i in ['When it comes to love']:  
            new_profile[i] = np.random.choice(combined[i], 1, p=p[i])
            
        elif i == 'Age':
            new_profile[i] = halfnorm.rvs(loc=18,scale=8, size=1).astype(int)
            
        else:
            new_profile[i] = list(np.random.choice(combined[i], size=(1,2), p=p[i]))
            
            new_profile[i] = new_profile[i].apply(lambda x: list(set(x.tolist())))
            

else:
    for i in new_profile.columns:
        new_profile[i] = st.selectbox(f"Enter your choice for {i}", combined)
        if i in ['When it comes to love']:  
            new_profile[i] = st.selectbox(f"Enter your choice for {i}", combined[i])
            
        elif i == 'Age':
            new_profile[i] = st.slider("What is your age?", 18, 50)
        else:

            options = st.multiselect(f"What is your preferred choice for {i}?", combined[i])
            
                                 

            new_profile.at[new_profile.index[0], i] = options
            
            new_profile[i] = new_profile[i].apply(lambda x: list(set(x)))
            
            
for col in df.columns:
    df[col] = df[col].apply(string_convert)
    
    new_profile[col] = new_profile[col].apply(string_convert)
            


st.write("-"*1000)
# st.write("Your profile:")
st.table(new_profile)

button = st.button("Click to find your Match!")

if button:    
    with st.spinner('Finding your Top Match'):

        df_v, input_df = vectorization(df, df.columns, new_profile)
        new_df = scaling(df_v, input_df)
        cluster = model.predict(new_df)
        top_df = top(cluster, new_df)  
        st.success("Found your Top Most Similar Profile!")    
        st.balloons()
        st.table(top_10_df)
        

        

    

