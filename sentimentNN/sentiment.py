import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sn

plt.style.use('ggplot')

import nltk
import os


df = pd.read_csv('/Users/steventussel/playingWithPandas/sentimentNN/Reviews.csv')

df = df.head(500) ##slimming down data set bc 1/2 million is a lot

#df['Score'].value_counts().sort_index().plot(kind='bar', title='whatever',figsize=(10,5))
#plt.show()

#NLTK
example = df['Text'][50]

#print(example)
#print(nltk.word_tokenize(example))

tokens = nltk.word_tokenize(example)

#part of speech tags
tagged = nltk.pos_tag(tokens)
#print(tagged)

#groups into chunks of text

entities = nltk.chunk.ne_chunk(tagged)
#entities.pprint()

#VADER
#remove stop words (and, the, a )
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()

#print(sia.polarity_scores('I  dont understand why this show is so  tererible. it  ruined my night.'))

# run polarity score on entire dataset

res = {}


for i, row in df.iterrows():
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)
#print(res)

vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')

#putting it with the original data set

# print(vaders)

# ax = sn.barplot(data=vaders, x='Score', y='compound')
# ax.set_title('whatever')
# plt.show()

# fig, axs =  plt.subplots(1,3,figsize=(15,5))
# sn.barplot(data=vaders, x='Score',y = 'pos', ax = axs[0])
# sn.barplot(data=vaders, x='Score',y = 'neu', ax = axs[1])
# sn.barplot(data=vaders, x='Score',y = 'neg', ax = axs[2])
# plt.show()


#roberta 

from transformers import AutoTokenizer 
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

print(example)



def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')

    output = model(**encoded_text)

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    print(scores)

    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict




# res = {}
# for i, row in df.iterrows():
#     try:
#         text = row['Text']
#         myid = row['Id']
#         vader_result = sia.polarity_scores(text)

#         roberta_result = polarity_scores_roberta(text)
#         both = {**vader_result, **roberta_result}
#         res[myid] = both
#     except RuntimeError:
#         print(f'Broke for id {myid}')




