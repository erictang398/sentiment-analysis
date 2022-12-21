import pickle
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

cv = CountVectorizer(ngram_range=(1,2))

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

input = "love this"
input = dataCleaner(input)

# convert the 1s in the matrix into a vector, .transform will produce the hits in array, .toarray converts to vector 
token = cv.transform([input]).toarray()
output = model.predict(token)
if (output):
    print("positive")
else:
    print("negative")