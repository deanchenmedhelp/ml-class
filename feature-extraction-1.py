# First attempt at feature extraction
# Leads to an error, can you tell why?

import pandas as pd
import numpy as np

df = pd.read_csv('tweets.csv')
target = df['is_there_an_emotion_directed_at_a_brand_or_product']
text = df['tweet_text']

#print(df.shape)
#print(df.describe)

from sklearn.feature_extraction.text import CountVectorizer

count_vect=CountVectorizer()
#count_vect.fit(text)

# creates structure of the spreadsheet                                          
# fit (sets up the columns, e.g. A, aardvark, apple, etc) and then              
# transform                                                                     
# count_vect.fit(text)                                                          

# try fit on subset, let say first 7 of the data if encountered error           
count_vect.fit(text[0:6])

#print(count_vect.vocabulary_.get(u'3g'))
