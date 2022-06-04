DROP TABLE IF EXISTS post_requests;
DROP TABLE IF EXISTS training_data_location;
DROP TABLE IF EXISTS ML_Training_Anomaly;
DROP TABLE IF EXISTS ML_Testing_Anomaly;
DROP TABLE IF EXISTS DL_Training_Anomaly;
DROP TABLE IF EXISTS DL_Testing_Anomaly;

/* Table for all incoming post requests */
CREATE TABLE post_requests (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  device TEXT NOT NULL,
  category TEXT NOT NULL,
  ml_type TEXT NOT NULL,
  monitors TEXT NOT NULL,
  behavior TEXT NOT NULL,
  training_path TEXT NOT NULL,
  arrival TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

/* A table with the location of the training data */
CREATE TABLE training_data_location(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  device TEXT NOT NULL,
  ml_type TEXT NOT NULL,
  location_path TEXT NOT NULL
);

/* A table for ML Anomaly Detection*/
CREATE TABLE ML_Training_Anomaly(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  device TEXT NOT NULL,
  feature TEXT NOT NULL,
  model TEXT NOT NULL,
  TNR FLOAT NOT NULL,
  train_time FLOAT NOT NULL
);

/* This table is for training */
CREATE TABLE ML_Testing_Anomaly(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  device TEXT NOT NULL,
  experiment TEXT NOT NULL,
  ransomware TEXT NOT NULL,
  feature TEXT NOT NULL,
  model TEXT NOT NULL,
  TPR FLOAT NOT NULL,
  test_time FLOAT NOT NULL
  );


/* A table for Anomaly Detection Deep Learning*/
CREATE TABLE DL_Training_Anomaly(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  device TEXT NOT NULL,
  feature TEXT NOT NULL,
  model TEXT NOT NULL,
  threshhold FLOAT NOT NULL,
  neurons TEXT NOT NULL,
  TNR FLOAT NOT NULL,
  train_time FLOAT NOT NULL
);

/* Table for testing */
CREATE TABLE DL_Testing_Anomaly(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  device TEXT NOT NULL,
  experiment TEXT NOT NULL,
  ransomware TEXT NOT NULL,
  feature TEXT NOT NULL,
  model TEXT NOT NULL,
  TPR FLOAT NOT NULL,
  test_time FLOAT NOT NULL
  );
