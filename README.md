# Disaster Response Pipeline Project

### Libraries needed:
Pandas
Numpy
matplotlib
Sci-kit Learn
Flask
SQL Alchemy
NLTK

### Files 
Apart from the files there are three folder the contents of which are
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
||- classifier.pkl  # saved model

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Demo of Web Application
Classification when message related to earthquake is processed
https://github.com/bhartgupta4/Disaster-Response-pipeline/blob/master/screenshot.PNG

The findings can be found here https://medium.com/@bhartlive12/building-a-disaster-response-application-26ccb13530a6

