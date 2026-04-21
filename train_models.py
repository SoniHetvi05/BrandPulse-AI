import pandas as pd
import numpy as np
import joblib
from utils import clean_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# 1. Load Data
df = pd.read_csv('mock_tweets.csv')
df['cleaned'] = df['text'].apply(clean_text)
y = pd.get_dummies(df['airline_sentiment']).values
y_labels = df['airline_sentiment']

# 2. Train Classical (TF-IDF + Logistic)
print("Training Classical Model...")
tfidf = TfidfVectorizer(max_features=2000)
X_tfidf = tfidf.fit_transform(df['cleaned'])
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_tfidf, y_labels, test_size=0.2)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_t, y_train_t)
joblib.dump(clf, 'models/classical_model.pkl')
joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')

# 3. Train Deep Learning (LSTM)
print("Training LSTM Model...")
max_words = 2000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['cleaned'])
X_seq = tokenizer.texts_to_sequences(df['cleaned'])
X_pad = pad_sequences(X_seq, maxlen=50)

X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_pad, y, test_size=0.2)

model = Sequential([
    Embedding(max_words, 128, input_length=50),
    LSTM(64, dropout=0.2),
    Dense(3, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_l, y_train_l, epochs=3, batch_size=32)
model.save('models/lstm_model.h5')
joblib.dump(tokenizer, 'models/tokenizer.pkl')

print("Success: All models saved in models/ directory.")