'''
The target of this module (train_classifier.py)(1) is to write a ML Pipeline:


* Loads data from the SQLite database
* Splits the dataset into training and test sets
* Builds a text processing and machine learning pipeline
* Trains and tunes a model using GridSearchCV
* Outputs results on the test set
* Exports the final model as a pickle file


(1) In project workspace 5.Project Workspace - ML pipeline name the module in the last point how train.py,but the file download in resources is named how train_classifier.py. In the project I going to keep the name train_classifier.py


[SQLAlchemy engine, DATABASE name:DisasterResponse, and TableName: disaster_response]
'''

# import libraries
import sys
import pandas as pd
import numpy as np
import re
import pickle
import joblib as jb
import nltk

nltk.download('wordnet') # download for lemmatization
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,TfidfTransformer
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.model_selection import GridSearchCV



import warnings

warnings.simplefilter('ignore')

##### LOAD DATA FROM DATABASE #####
def load_data(database_filepath):

    """
    Load DATA from DataBAse in Dataframe
    
    Arguments:
        database_filepath -> Path to the Database
        
    Output:
       X -> Contains dataframe with features
       y -> Contains dataframe with labels
       category_names->     Contains list with category_nmes
    """
    
    # load data from database
    # DataBase: Disaster_response_pipeline
    # table: disaster_responses
   
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disaster_response',engine)
    



    X = df['message']
    y = df[['offer','aid_related', 'medical_help', 'medical_products', 'search_and_rescue','security', 'military', 'water', 'food', 'shelter','clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid','infrastructure_related', 'transport', 'buildings', 'electricity','tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure','weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold','other_weather', 'direct_report']]

       
    category_names = y.columns
    return X, y, category_names
  

##### Transform in tokens corpus #####
def tokenize(text):
    """
    Tokenize text
    Arguments:
        text -> Contains the info to transform in tokens
    Output:
       tokens -> Contains tokens
              
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    
    stop_words = stopwords.words("english")
    #  normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    text = re.sub("<", " ", text)
    text = re.sub(">", " ", text)
    #print(text)
  
                  
    # Anything that isn't A through Z or 0 through 9 will be replaced by a space
    text = text.lower()
    #print(text)
    
    #  tokenize text
    tokens = text.split()
    #print(text)
    
       
    # lemmatize and remove stop words
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [WordNetLemmatizer().lemmatize(w) for w in tokens]
        
    return tokens

##### BUILD a CUSTOM  TRANSFORMER #####class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    '''
    Starting Verb Extractor class
    Custom transformer which will extract the starting verb of a sentence

    
    '''

# Build a custom transformer which will extract the starting verb of a sentence
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class
    
    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    # Given it is a tranformer we can return the self
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

##### BUILD MODEL #####
def build_model():
   '''
   Build a Machine Learning  Pipeline with TFIDF, starting verb, and a MultiOutputClassifier.
       
   Output:
       cv -> GridSearchCV. Transforms  data, creates the  model_selection
   '''

   pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
   parameters = {
        'vect__stop_words':['english'],
        'vect__ngram_range':[(1,2)],
        'vect__max_features':[40000],
        'clf__estimator__learning_rate':[0.75, 1.0],
        'clf__estimator__n_estimators':[50, 75]
    }
    
   cv = GridSearchCV(pipeline, parameters, scoring='recall_macro', cv=3)

   return cv



##### EVALUATE MODEL #####

def evaluate_model(model, X_test, Y_test, category_names):
    '''
   Evaluate the model
   
   Arguments:
    model -> Contains the model
    X_test -> Contains dataframe with features dataset
    Y_test -> Contains dataframe with labels
    category_names -> List containing category names.
         
   '''

    # Get the prediction values from the grid search cross validator
         
   # Use model to predict
    y_prediction_test  = model.predict(X_test)
    #y_prediction_train = model.predict(X_train)
    
    
    
   
   


    # Print classification report on test data
    print(classification_report(Y_test.values, y_prediction_test, target_names=Y_test.columns.values))

    # Print classification report on test data
    #print(classification_report(Y_test.values, y_prediction_test, target_names=y.columns.values))
    


##### SAVE THE MODEL #####

def save_model(model, model_filepath):
    '''
    Evaluate the model
    Arguments:
        model -> Contains the model
        model_filepath -> Contains dataframe with features dataset
    Output:
       file pickle->   save the model in a pickle file in model_filepath
    '''
        
    pickle.dump(model, open(model_filepath, "wb"))

    
def main():
    '''
    To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        
        print('[--1--] Loading data...\n    DATABASE: {}'.format(database_filepath))
        
        X, Y, category_names = load_data(database_filepath)
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('[--2--] Building model...')
        model = build_model()
        
        print('[--3--] Training model...')
        model.fit(X_train, Y_train)
        
        print('[--4--]Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('[--5--]Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('[--DONE--]Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
