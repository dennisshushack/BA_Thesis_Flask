# Repository Flask (Data Analysis Application)

This is the repository for the Data Analysis Application and is part of a thesis. The Monitor Controller repo can be found here: https://github.com/dennisshushack/BA_Thesis_PI.

## Structure of the repository:
The entire Flask appication is in the `app` Folder. It is structured in a Model View Controller fashion:
1. **Controllers**: Contain the Endpoints for the requests:
   * `controllers/AppController.py` is for the endpoints concerning the sensor.
   * `controllers/FronendController.py` is as the name suggests for the GUI.
