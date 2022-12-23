import numpy as np
import pandas as pd
import nltk
import pickle
from bs4 import BeautifulSoup

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

nltk.download('stopwords')

df = pd.read_csv('movieData.csv')

# select top n rows from each label
def get_top_n_rows_per_group(data, n, columodel):
    grouped_data = data.groupby(columodel)
    top_n_rows = grouped_data.apply(lambda x: x.sort_values('label', ascending=False).head(n))
    return top_n_rows.reset_index(drop=True)

# Get the top 10000 rows for each group
data = get_top_n_rows_per_group(df, 10000, 'label')

# transform into numpy array
textData = np.array(data["text"])
labelData = np.array(data["label"])

# turn into a python list
textData = textData.tolist()

# declaring some objects for preprocessing
tokenizer = RegexpTokenizer(r'\w+')
en_stopwords = set(stopwords.words('english'))
ps = PorterStemmer()

# data preprocessing function
def dataCleaner(text):
  soup = BeautifulSoup(text, 'html.parser')

  text = soup.get_text()

  text = text.lower()

  tokens = tokenizer.tokenize(text)

  newTokens = [word for word in tokens if word not in en_stopwords]

  stemmedTokens = [ps.stem(token) for token in newTokens]

  cleanedText = " ".join(stemmedTokens)

  return cleanedText

textData = [dataCleaner(i) for i in textData]
textData = np.array(textData)

# vectorizing text
cv = CountVectorizer(ngram_range=(1,2))
vectorizedData = cv.fit_transform(textData)

X_train, X_test, y_train, y_test = train_test_split(vectorizedData, labelData, test_size=0.2, random_state=42)

# training the model
model = MultinomialNB()
model.fit(X_train, y_train)

# testing with a phrase
input = "this is awesome!" # testing phrase
input = dataCleaner(input)

# convert the 1s in the matrix into a vector, .transform will produce the hits in array, .toarray converts to vector 
token = cv.transform([input]).toarray()
output = model.predict(token)
if (output):
    print("positive")
else:
    print("negative")
