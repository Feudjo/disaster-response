# Disaster Response Pipeline Project
### Introduction:
Disaster response organizations receive millions of unstructured text messages via social media after a disaster.
This information needs to be categorized and assigned to the responsible department in the organization.
We build a machine learning model that assigns a label to an incoming message.


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
