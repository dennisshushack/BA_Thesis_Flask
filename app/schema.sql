DROP TABLE IF EXISTS post_requests;
DROP TABLE IF EXISTS training_data_location;
DROP TABLE IF EXISTS ML_Training_Anomaly;
DROP TABLE IF EXISTS ML_Testing_Anomaly;
DROP TABLE IF EXISTS DL_Training_Anomaly;
DROP TABLE IF EXISTS DL_Testing_Anomaly;
DROP TABLE IF EXISTS live_anomaly;

/* A table with the location of the training data */
CREATE TABLE training_data_location(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  device TEXT NOT NULL,
  ml_type TEXT NOT NULL,
  location_path TEXT NOT NULL
);

/* Table for training AD */
CREATE TABLE ML_Training_Anomaly(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  device TEXT NOT NULL,
  feature TEXT NOT NULL,
  model TEXT NOT NULL,
  TNR FLOAT NOT NULL,
  train_time FLOAT NOT NULL,
  test_time FLOAT NOT NULL
);

/* Table for testing AD */
CREATE TABLE ML_Testing_Anomaly(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  trainings_id INTEGER NOT NULL,
  device TEXT NOT NULL,
  experiment TEXT NOT NULL,
  ransomware TEXT NOT NULL,
  feature TEXT NOT NULL,
  model TEXT NOT NULL,
  TPR FLOAT NOT NULL,
  FOREIGN KEY(trainings_id) REFERENCES ML_Training_Anomaly(id)
  );


/* table for training DL AD*/
CREATE TABLE DL_Training_Anomaly(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  device TEXT NOT NULL,
  feature TEXT NOT NULL,
  model TEXT NOT NULL,
  STD_lower FLOAT NOT NULL,
  STD_upper FLOAT NOT NULL,
  IQR_lower FLOAT NOT NULL,
  IQR_upper FLOAT NOT NULL,
  neurons TEXT NOT NULL,
  TNR_STD FLOAT NOT NULL,
  TNR_IQR FLOAT NOT NULL,
  train_time FLOAT NOT NULL,
  test_time FLOAT NOT NULL
);

/* Table for testing DL AD */
CREATE TABLE DL_Testing_Anomaly(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  trainings_id INTEGER NOT NULL,
  device TEXT NOT NULL,
  experiment TEXT NOT NULL,
  ransomware TEXT NOT NULL,
  feature TEXT NOT NULL,
  model TEXT NOT NULL,
  TPR_STD FLOAT NOT NULL,
  TPR_IQR FLOAT NOT NULL,
  FOREIGN KEY(trainings_id) REFERENCES DL_Training_Anomaly(id)
  );

/* Table for live evaluation */
CREATE TABLE live_anomaly(
id INTEGER PRIMARY KEY AUTOINCREMENT,
device TEXT NOT NULL,
ml_type TEXT NOT NULL,
algo TEXT NOT NULL,
feature TEXT NOT NULL,
time_stamp INTEGER NOT NULL,
anomaly_class INTEGER NOT NULL,
testing_time FLOAT NOT NULL
);
