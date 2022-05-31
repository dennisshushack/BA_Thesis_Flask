DROP TABLE IF EXISTS post_requests;
DROP TABLE IF EXISTS training_data_location;
DROP TABLE IF EXISTS ML_Anomaly;
DROP TABLE IF EXISTS DL_Anomaly;



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
CREATE TABLE ML_Anomaly(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  device TEXT NOT NULL,
  feature TEXT NOT NULL,
  model TEXT NOT NULL,
  TNR FLOAT NOT NULL,
  train_time FLOAT NOT NULL,
  TPR_ransom1 FLOAT DEFAULT NULL,
  test_time_ransom1 FLOAT DEFAULT NULL,
  TPR_ransom2 FLOAT DEFAULT NULL,
  test_time_ransom2 FLOAT DEFAULT NULL,
  TPR_ransom3 FLOAT DEFAULT NULL,
  test_time_ransom3 FLOAT DEFAULT NULL
);

/* A table for Anomaly Detection Deep Learning*/
CREATE TABLE DL_Anomaly(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  device TEXT NOT NULL,
  feature TEXT NOT NULL,
  model TEXT NOT NULL,
  threshhold FLOAT NOT NULL,
  neurons TEXT NOT NULL,
  TNR FLOAT NOT NULL,
  train_time FLOAT NOT NULL,
  TPR_ransom1 FLOAT DEFAULT NULL,
  test_time_ransom1 FLOAT DEFAULT NULL,
  TPR_ransom2 FLOAT DEFAULT NULL,
  test_time_ransom2 FLOAT DEFAULT NULL,
  TPR_ransom3 FLOAT DEFAULT NULL,
  test_time_ransom3 FLOAT DEFAULT NULL
);


