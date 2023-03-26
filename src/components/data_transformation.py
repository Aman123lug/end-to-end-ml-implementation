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
@dataclass
class DatatranformationConfig:
    transformation_path = os.path.join("artifacts", "transformation.pkl")
    train_data_path: str= os.path.join("notebooks\data", "train.csv")
    test_data_path: str= os.path.join("notebooks\data", "test.csv")
    
    
    
tokenizer=RegexpTokenizer("\w+")
sw = set(stopwords.words("english"))
ps=PorterStemmer()

def getStem(review):
    review = review.lower()
    tokens = tokenizer.tokenize(review)
    removed_stopwords = [w for w in tokens if w not in sw]
    stemmed_words = [ps.stem(token) for token in removed_stopwords]
    clean_review = ' '.join(stemmed_words)
    return clean_review
class Datatranform:
    def __init__(self):
        self.datatransformconfig = DatatranformationConfig()
        
        
    def data_transformer(self, data_path):
        try:
            data = pd.read_csv(data_path)
            data = getStem(data)
            logging.info('train_test_split')
            
            X_train ,X_test = train_test_split(data, random_state=42, test_size=0.20)
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)
            logging.info('file saved')
            X_train.to_csv(self.datatransformconfig.train_data_path)
            X_test.to_csv(self.datatransformconfig.test_data_path)
            
            return self.datatransformconfig.train_data_path, self.datatransformconfig.train_data_path
            
        except Exception as e:
            CustomErrorHandler(e, sys)
        
            
            
            
            
    def initailize_data_transform(self, x_train_path, x_test_path):
        logging.info("Count vectorizers")
        try:
            X_train = pd.read_csv(self.datatransformconfig.train_data_path)
            X_test = pd.read_csv(self.datatransformconfig.train_data_path)
            
            cv = CountVectorizer()
            X_train_new = cv.transform(X_train)
            X_test_new = cv.transform(X_test)
            
            X_train, y_train = X_train_new
            X_test, y_test = X_test_new
            
            
            logging.info("model training started")
            
            model = MultinomialNB()
            model.fit(X_train, y_train)
            ypred = model.predict(X_test)
            logging.info("model trained")
            print(accuracy_score(ypred, y_test))
            
            
        except Exception as e:
            CustomErrorHandler(e, sys)
                        
if __name__ == "__main__":
    obj = Datatranform()
    x_train_path, x_test_path = obj.data_transformer("notebooks\data\spamsms.csv")
    obj.initailize_data_transform(x_train_path, x_test_path)
    
            
            
            
            

          