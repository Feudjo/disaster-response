import sys
import pickle
import re
import pandas as pd
from sqlalchemy import create_engine
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report





def load_data(database_filepath):
    """

    :param database_filepath:
    :type database_filepath:str
    :return:
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponse.db', con=engine)
    return df

def tokenize(text):
    """
    Text preprocessor.
    :param text:
    :type text: str
    :return:
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

    return tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer())
        ('rfc', MultiOutputClassifier(RandomForestClassifier()))
    ], verbose=True)
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """

    :param model:
    :param X_test:
    :type X_test: np.array
    :param Y_test:
    :type Y_test: np.array
    :param category_names:
    :type category_names: list
    :return:
    """
    y_pred = model.predict(X_test)
    for i, c in enumerate(category_names):
        print(classification_report(Y_test[c], y_pred[c], target_names=[category_names[i]]))



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