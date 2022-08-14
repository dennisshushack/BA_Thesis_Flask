# Repository Flask (Data Analysis Application)

This is the repository for the Data Analysis Application and is part of a thesis. The Monitor Controller repo can be found here: https://github.com/dennisshushack/BA_Thesis_PI. It is suggested to install the Monitor Controller, before following this installation guide.

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
  * `app/schema.sql`: Defines the database tables

3. The **View** is the Graphical User Interface. It encompasses the `app/templates` and the `app/static` folders.

## Installing the Flask Application

### Prerequsites:
A computer running Linux or a BSD-based distribution (i.e. Ubuntu, Arch, MAC OS) connected to LAN.
A modern Python version must be installed on that computer (above 3.5). It is also helpfull to install a SQLlite database viewer, such as DB Browser for SQLite (https://sqlitebrowser.org/)

### Installation:
Please follow these commands to install the Flask Application (example Ubuntu):
```
apt install python3-venv
apt-get install git
git clone https://github.com/dennisshushack/BA_Thesis_Flask.git
cd BA_Thesis_Flask
python3 -m venv env
source env/bin/activate
pip install flask numpy pandas pyod tensorflow
```
To run the application use:
```
source env/bin/activate
export FLASK_APP=app
export FLASK_ENV=development
flask init-db
flask run --host=serverip
```
If you do not know the IP of your machine, use the `ifconfig` command. It is extremely important to have the Application running on your IP address, as networking with the sensor would otherwise become fairly difficult. This assumes of course, that you are testing this on your LAN Network. Furthermore, after executing these commands a new folder will appear called instance. It houses the database file, that can be viewed with the SQLite viewer.

### Accessing the Front-End:
You can access the frontend by opening a Browser on `http://ipFlask:Port/live`. It is protected by Basic Authentication. Use the default username = admin and pw = admin to access it. It displays only the live monitoring sessions : ![Screenshot from 2022-08-14 16-43-29](https://user-images.githubusercontent.com/24684973/184542920-8d0a018b-9afe-4b91-8c1f-9729975ee8e1.png)







