# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 19:36:13 2020

@author: DELL
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt 

data=pd.read_csv("E:\\assignment\\ann 16 tiwwter sentient analysis\\train_E6oV3lV.csv")
data.head()
data.tail()
data['label'].unique()
data.shape
tdata=pd.read_csv("E:\\assignment\\ann 16 tiwwter sentient analysis\\test_tweets_anuFYb8.csv")
tdata.shape
tdata.head()
tdata.tail()
fdata=data.append(tdata,sort=False)
fdata.tail()
fdata.reset_index(inplace=True)
fdata.tail()
import re
import nltk
nltk.download('stopwords') #contain irrelvant words  & ava in diff lang
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = [] #corpus collection of strings
for i in range(0,49159):
    review = re.sub('[^a-zA-Z]', ' ', str(fdata['tweet'][i])) #remove  except a-z & A-Z & create space b/w words
    review = review.lower() #capital to lower
    review = review.split()  #sentence to words
    ps = PorterStemmer()  # loved to love (diff kind of same word into standard word)
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] #take words not in stopward
    review = ' '.join(review) #again to string sep by space
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 8000)
X = cv.fit_transform(corpus).toarray()
y = data['label']
X.shape
xtrain=X[0:31962]
ytrain=y[0:31962]
xtest=X[31962:]
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(xtrain, ytrain)

y_pred = classifier.predict(xtest)

results = np.array(y_pred)
results = pd.Series(results,name="pred")

submission = pd.concat([pd.Series(range(1,36001),name = "ImageId"),results],axis = 1)

submission.to_csv("rf100.csv",index=False)
