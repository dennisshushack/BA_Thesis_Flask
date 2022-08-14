# Repository Flask (Data Analysis Application)

This is the repository for the Data Analysis Application and is part of a thesis. The Monitor Controller repo can be found here: https://github.com/dennisshushack/BA_Thesis_PI.

If you have any questions related to the installation or execution of either the Flask Application or the Monitor Controller, you can write me an email:
dennis.shushack@uzh.ch

## Structure of the repository:
The entire Flask application is in the `app` Folder. It is structured in a Model View Controller fashion:
1. **Controllers**: Contain the Endpoints for the requests:
   * `app/controllers/AppController.py` is for the endpoints concerning the sensor.
   * `app/controllers/FronendController.py` is as the name suggests for the GUI. User requests are handled here.
   
2. The Controllers forward the data from the requests to the **Model**. The Model handles the application's logic, including data processing, ML/DL, and database-related work. The following folders contain the main parts of the **Model**:
  * `àpp/services` contains so-called service classes. They perform the main code logic: data pre-processing, ML/DL anomaly, and classification training/evaluation.
  * `àpp/database/dbqueries`: Are all database queries (SQLite) for the Flask Application 

3. The **View** is the Graphical User Interface. It encompasses the `app/templates` and the `app/static` folders.

## Installing the Flask Application

# Prerequsites:
