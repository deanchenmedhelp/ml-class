import pandas as pd
import numpy as np

# data frame
df = pd.read_csv('tweets.csv')

# data vector from the column matching given title
target = df['is_there_an_emotion_directed_at_a_brand_or_product']
text = df['tweet_text']


#print(target[9])

# index starts at 0
# print(text[9])
print target[0:5]
