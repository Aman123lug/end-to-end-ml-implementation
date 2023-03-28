from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from sklearn.naive_bayes import MultinomialNB
import sys
from src.exception import CustomErrorHandler
from src.logger import logging
import os
from sklearn.metrics import accuracy_score
import pickle
import warnings
# warnings.ignore("warnings")
@dataclass
class DatatranformationConfig:
    transformation_path = os.path.join("artifacts", "transformation.pkl")
    train_data_path: str= os.path.join("notebooks/data", "train.csv")
    test_data_path: str= os.path.join("notebooks/data", "test.csv")
    
    
    
tokenizer=RegexpTokenizer("\w+")
sw = set(stopwords.words("english"))
ps=PorterStemmer()

def getStem(review):
    review = str(review)
    review = review.lower()
    tokens = tokenizer.tokenize(review)
    removed_stopwords = [w for w in tokens if w not in sw]
    stemmed_words = [ps.stem(token) for token in removed_stopwords]
    clean_review = ' '.join(stemmed_words)
    return clean_review

def getDoc(document):
    d = []
    for doc in document:
        d.append(getStem(doc))
    return d

            
def DataTranformForTest(text):
    cv = CountVectorizer()
    X = getDoc(text)
    new_x = cv.fit_transform(X)
    preprocess_data = new_x.toarray()
    
    return preprocess_data
class Datatransform:
    def __init__(self):
        self.datatransformconfig = DatatranformationConfig()
        
        
    def data_transformer(self, data_path):
        try:
            data = pd.read_csv("notebooks\\data\\spamsms.csv", encoding='ISO-8859-1')
            data= data[["v1","v2"]]

            y = data["v1"]
            X = data["v2"]

            X = getDoc(X)

            logging.info("train_test_split")
            X_train ,X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.20)

            y_train = pd.DataFrame(y_train)
            y_test = pd.DataFrame(y_test)

            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)

            X_train = pd.concat([X_train, y_train], axis=1)
            X_test = pd.concat([X_test, y_test], axis=1)
            X_train.dropna(inplace=True)
            X_test.dropna(inplace=True)
            logging.info("Train_test_file saved ")
            X_train.to_csv("notebooks\\data\\train.csv")
            X_test.to_csv("notebooks\\data\\test.csv")

            
            
            
        except Exception as e:
            CustomErrorHandler(e, sys)
        
            
    def initailize_data_transform(self, x_train_path, x_test_path):
        logging.info("Count vectorizers")
        try:
            X_train = pd.read_csv(self.datatransformconfig.train_data_path)
            X_test = pd.read_csv(self.datatransformconfig.train_data_path)
            
            cv = CountVectorizer()
            X_train.rename(columns={"0":"text","v1":"label"}, inplace=True)
            X_train = X_train[["text", "label"]]
            # xtrain ready
            X = X_train[["text"]] 
            y = X_train[["label"]] 

            X = getDoc(X_train["text"])
            new_x = cv.fit_transform(X)
            X_train = new_x.toarray()


            lb = LabelEncoder()
            y_train = lb.fit_transform(y)
            logging.info("model training started")
            #  model training
            model = MultinomialNB()
            model.fit(X_train, y_train)
            filename = 'saved_ml_model/model.pkl'
            pickle.dump(model, open(filename, 'wb'))
            logging.info("model saved")
            
            
        except Exception as e:
            CustomErrorHandler(e, sys)

    
        
            
                     
if __name__ == "__main__":
    new_obj = DatatranformationConfig()
    
    obj = Datatransform()
    
    obj.data_transformer("notebooks/data/spamsms.csv")
    obj.initailize_data_transform("notebooks/data/spamsms.csv", "notebooks/data/spamsms.csv")
    
            
            

            

          