import pandas
import numpy
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense,Embedding,LSTM
from tensorflow.keras.models import load_model,Sequential

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

maxlength=1000
df=pandas.read_csv("data/language_identification.csv")
encoder=LabelEncoder()
lemmatizer=WordNetLemmatizer()
tokenizer=Tokenizer(oov_token="UNK")

print(df.describe())
print(df.head())

x=df["Text"]
y=df["language"]

y=encoder.fit_transform(y)
labels=encoder.classes_
print(labels)

y=to_categorical(y)
print(y)

tokenizer.fit_on_texts(x)
vocabulary_size=len(tokenizer.word_index)+1
print(vocabulary_size)

def process_texts(texts):
  to_return=tokenizer.texts_to_sequences(texts)
  return pad_sequences(to_return,maxlength,truncating="post")

x=process_texts(x)

x_train,x_test,y_train,y_test=train_test_split(x,y)

model=Sequential([
    Embedding(vocabulary_size,64,input_length=maxlength),
    LSTM(64),
    Dense(32),
    Dense(len(labels),"softmax")
])
model.compile("adam","categorical_crossentropy","accuracy")
model.fit(x_train,y_train,epochs=2,validation_data=(x_test,y_test))

test_data=["Today i will talk about SOLID principles. SOLID stands for: Single Responsibility, Open-closed, Liskov's Substitution, Interface Segregation and Dependency Inversion. Basically these 5 principles help us develop easily changeable applications and projects"]
test_data=process_texts(test_data)
result=model.predict(test_data)
print(labels[numpy.argmax(result)])