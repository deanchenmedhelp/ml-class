import pandas as pd
import numpy as np

df = pd.read_csv('tweets.csv')
target = df['is_there_an_emotion_directed_at_a_brand_or_product']
text = df['tweet_text']

fixed_text = text[pd.notnull(text)]
fixed_target = target[pd.notnull(text)]

from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
count_vect.fit(fixed_text)

# turns the text into a sparse matrix
counts = count_vect.transform(fixed_text)

# prints first 2 rows
print(fixed_text[0:2])

# prints first 2 rows, omitting 0s in the sparse matrix
print(counts[0:2])

# vocabulary_ is a dictionary                                                   
# finds the key where value is 430 in the dictionary                            
# mydict = count_vect.vocabulary_
# print mydict.keys()[mydict.values().index(430)]
# or 
# print(count_vect.get_feature_names()[430])                                 

# some other fun things to try
#print(fixed_text[0])
#print(count_vect.transform(["cerulean"]))
