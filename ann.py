# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 19:55:06 2020

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
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 4000, kernel_initializer = 'he_uniform', activation = 'relu', input_dim = 8000))
classifier.add(Dropout(0.4))
# Adding the second hidden layer

# Adding the second hidden layer
classifier.add(Dense(units = 4000, kernel_initializer = 'he_uniform', activation = 'relu'))
classifier.add(Dropout(0.4))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'he_uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



classifier.fit(xtrain, ytrain, batch_size = 128, epochs = 6)

y_pred = classifier.predict(xtest)
y_pred = np.concatenate(y_pred)
res=[]
y=0.5
for i in y_pred :
     if  i>0.5:
         res.append(1)
     else:
         res.append(0)
results = np.array(res)

results = pd.Series(results,name="pred")

submission = pd.concat([pd.Series(range(1,3600),name = "ImageId"),results],axis = 1)

submission.to_csv("ann 3 3000 layers e-10 adam 5000token .csv",index=False)
