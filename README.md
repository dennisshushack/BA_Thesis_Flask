# Repository Flask (Data Analysis Application)

This is the repository for the Data Analysis Application and is part of a thesis. The Monitor Controller repo can be found here: https://github.com/dennisshushack/BA_Thesis_PI.

## Structure of the repository:
The entire Flask appication is in the `app` Folder. It is structured in a Model View Controller fashion:
1. **Controllers**: Contain the Endpoints for the requests:
   * `app/controllers/AppController.py` is for the endpoints concerning the sensor.
   * `app/controllers/FronendController.py` is as the name suggests for the GUI. User requests are handeled here.
   
2. The Controllers forwards the data from the requests to the **Model**. The Model does the main data processing, including the ML/DL and database related work. The following folders contain the main logic of the **Model**:
  * `àpp/services` contains so called service classes. There perform the main code logic: data pre-processing, ML/DL anomaly and classification training/evaluation.
  * `àpp/database/dbqueries`: Are all database queries (SQLite) for the Flask Application 

3. The **View** is the Graphical User Interface. It encompases the `app/templates` and the `app/static` folders.
