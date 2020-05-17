import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import re
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
import os
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.ensemble import RandomForestClassifier

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support,classification_report
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline,FeatureUnion
import pickle

def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    table_name = database_filepath[:-3]
    table=table_name.split('/')
    table=str(table[1])
    df = pd.read_sql_table('DisasterMessages',con=engine)
    X = df['message']
    Y = df[df.columns[4:]]
    category_names = Y.columns
    return X,Y,category_names


def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_url=re.findall(url_regex,text)
    for url in detected_url:
        text=text.replace(url,"urlplaceholder")
    text=re.sub(r"[^a-zA-Z0-9]"," ",text)
    text=word_tokenize(text)
    lemmatizer=WordNetLemmatizer()
    clean_tokens=[]
    for tok in text:
        token=lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(token)
    return clean_tokens


    
def build_model():
    clsi = MultiOutputClassifier(RandomForestClassifier())

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', clsi)
        ])

    parameters = {'clf__estimator__max_depth': [None,10,50],
                  'tfidf__smooth_idf':[True,False],
              'clf__estimator__min_samples_leaf':[1, 5, 20]}

    cv = GridSearchCV(pipeline, parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n", classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
        print('Accuracy of %25s: %.2f' %(category_names[i], accuracy_score(Y_test.iloc[:, i].values, Y_pred[:,i])))

    


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()