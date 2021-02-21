import sys
import pickle
import re
import pandas as pd
from sqlalchemy import create_engine
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report





def load_data(database_filepath):
    """Reads data from database in database_filepath and returns a 3-turple.

    :param database_filepath:
    :type database_filepath:str
    :return:(features, labels, categories_name)
    :rtype: tuple
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster_table', con=engine)
    X = df['message']
    y = df.drop(['message', 'original', 'id', 'genre'], axis=1)
    return X, y, y.columns.to_list()

def tokenize(text):
    """ Text preprocessor. Returns cleaned, lemmatized tokens of text.

    :param text:
    :type text: str
    :rtype: list
    :return: list of clean tokens
    """
    stopword = stopwords.words('english')

    #Detect and remove urls, punctuation
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    text = re.sub("[^a-zA-Z0-9]+", ' ', text)

    # Lemmatize text while removing spaces at the beginning and end of strings
    tokens_list = []
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    for word in tokens:
        if word not in stopword:
            tokens_list.append(lemmatizer.lemmatize(word, pos='v').strip())

    return tokens_list


def build_model():
    """Returns a GridSearchCv object that performs hyperparameter optimization of the pipeline.
    :return: GridSearchCV
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer(norm='l1')),
        ('rfc', MultiOutputClassifier(RandomForestClassifier(n_jobs=2)))
    ], verbose=True)


    parameters = {
        'tfidf__use_idf': (True, False),
        'rfc__estimator__n_estimators': [50, 100, 200]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Prints classification report (precision, recall) for each label in category_names.

    :param model:
    :param X_test:
    :param Y_test:
    :param category_names:
    :type category_names: list
    :return: None
    """

    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=category_names)
    for i, c in enumerate(category_names):
       print(classification_report(Y_test[c], y_pred[c]))



def save_model(model, model_filepath):
    """Saves model

    :param model:
    :type model: sklear.pipeline.Pipeline
    :param model_filepath:
    :type model_filepath: str
    :return: None
    """
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