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

#print(data.head())

def get_top_n_rows_per_group(data, n, columodel):
    # Group the data by the columodel
    grouped_data = data.groupby(columodel)
    # For each group, get the top N rows
    top_n_rows = grouped_data.apply(lambda x: x.sort_values('label', ascending=False).head(n))
    # Reset the index of the resulting DataFrame
    return top_n_rows.reset_index(drop=True)

# Get the top 1500 rows for each group
data = get_top_n_rows_per_group(df, 10000, 'label')

# print(data)

# label_counts = data['label'].value_counts()

# print(label_counts)

textData = np.array(data["text"])
labelData = np.array(data["label"])

textData = textData.tolist()

# X_train, X_test, y_train, y_test = train_test_split(textData, labelData, test_size=0.2, random_state=42)

# X_train_arr = X_train.tolist()
# X_test_arr = X_test.tolist()
# y_train_arr = y_train.tolist()
# y_test_arr = y_test.tolist()

# X_train_arr = X_train_arr[0:2000]
# X_test_arr = X_test_arr[0:2000]
# y_train_arr = y_train_arr[0:2000]
# y_test_arr = y_test_arr[0:2000]

tokenizer = RegexpTokenizer(r'\w+')
en_stopwords = set(stopwords.words('english'))
ps = PorterStemmer()

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

cv = CountVectorizer(ngram_range=(1,2))
vectorizedData = cv.fit_transform(textData)
# X_train_clean = [dataCleaner(i) for i in X_train_arr]
# X_test_clean = [dataCleaner(j) for j in X_test_arr]

X_train, X_test, y_train, y_test = train_test_split(vectorizedData, labelData, test_size=0.2, random_state=42)

# X_train_arr = X_train.tolist()
# X_test_arr = X_test.tolist()
# y_train_arr = y_train.tolist()
# y_test_arr = y_test.tolist()

# cv = CountVectorizer(ngram_range=(1,2))
# X_vec = cv.fit_transform(X_train_clean) #.toarray()
# X_test_vec = cv.fit_transform(X_test_clean) #.toarray()

model = MultinomialNB()
model.fit(X_train, y_train)

# print(model.score(X_test, y_test))
#print(model.score(X_vec, y_train_arr))

input = "love this" # testing phrase
input = dataCleaner(input)

# convert the 1s in the matrix into a vector, .transform will produce the hits in array, .toarray converts to vector 
token = cv.transform([input]).toarray()
output = model.predict(token)
if (output):
    print("positive")
else:
    print("negative")

# with open('model.pkl', 'wb') as f:
#     pickle.dump(model, f)
