import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load two Dataframes. One From messages_filepath, the other from Categories_filepath.
    Merges both dataframes.


    :param messages_filepath
    :type messages_filepath: str
    :param categories_filepath
    :type categories_filepath: str
    :return: pd.DataFrame
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df_merge = pd.merge(messages, categories, on=['id'])

    return df_merge


def clean_data(df):
    """
    Data transformation

    :param df: dframe to clean
    :type pd.DataFrame
    :return: dframe
    """
    # Create different category columns
    df_categories = df.categories.str.split(';', expand=True)

    # Extract a List of new column names for categories
    row = df_categories.iloc[0]
    category_name = row.apply(lambda x: x[:-2])

    # Rename categories columns
    df_categories.columns = category_name

    # binarize category values
    for column in df_categories:
        # set each value to be the last character of the string
        df_categories[column] = df_categories[column].astype(str).str[-1]

        # convert column from string to numeric
        df_categories[column] = df_categories[column].astype('int32')

    df.drop('categories', axis=1, inplace=True)  # drop the original categories column from `df`

    df = pd.concat([df, df_categories], axis=1)  # concatenate df with new categories dframe.
    df.drop_duplicates(inplace=True)  # drop duplicates
    return df


def save_data(df, database_filename):
    """

    :param df:
    :type pd.DataFrame
    :param database_filename:
    :type  database_filename: str
    :return:
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster_table', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
