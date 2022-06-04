class dbqueries:

    @staticmethod
    def insert_into_post_requests(db_connection, device, category, ml_type, monitors, behavior, training_path):
        """
        Inserts the data into the post_requests table.
        :input: db_connection: database connection, description: description of the request, 
        category: category of the request, type: normal, ransom1, ransom2, path: path of the data
        device: an indicator for the device that made the request
        """
        cursor = db_connection.cursor()
        cursor.execute("""
            INSERT INTO post_requests (device, category, ml_type, monitors, behavior, training_path)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (device, category, ml_type, monitors, behavior, training_path))
        db_connection.commit()
        cursor.close()

    @staticmethod
    def get_post_requests(db_connection):
        """
        Gets all the post requests from the database.
        :input: db_connection: database connection
        :return: all the post requests
        """
        cursor = db_connection.cursor()
        cursor.execute("""
            SELECT * FROM post_requests
        """)
        post_requests = cursor.fetchall()
        cursor.close()
        return post_requests
    
    @staticmethod
    def insert_training_data_location(db_connection, device, ml_type, path):
        """
        Inserts the location of the training data into the database.
        :input: db_connection: database connection, device: device that made the request, monitor: 
        monitor that made the request, path: path of the data
        """
        cursor = db_connection.cursor()
        cursor.execute("""
            INSERT INTO training_data_location (device, ml_type, location_path)
            VALUES (?, ?, ?)
        """, (device, ml_type, path))
        db_connection.commit()
        cursor.close()

    @staticmethod
    def get_training_data_location(db_connection, device, ml_type):
        """
        Gets the location of the training data from the database.
        :input: db_connection: database connection, device: device that made the request, monitor: 
        monitor that is being used 
        :return: the location of the training data
        """
        cursor = db_connection.cursor()
        cursor.execute("""
            SELECT location_path FROM training_data_location
            WHERE device = ? AND ml_type = ?
        """, (device, ml_type))
        location = cursor.fetchone()
        cursor.close()
        if location is None:
            return None
        return location[0]
    
    @staticmethod
    def create_ml_anomaly(db_connection, device, feature, model, TNR, train_time):
        """
        Creates a new entry in the ml_dl_anomaly table.
        :input: db_connection: database connection, device: device that made the request, feature: 
        feature that is being used, model: model that is being used, TNR: TNR of the model, train_time:
        time it took to train the model
        """
        cursor = db_connection.cursor()
        cursor.execute("""
            INSERT INTO ML_Training_Anomaly (device, feature, model, TNR, train_time)
            VALUES (?, ?, ?, ?, ?)
        """, (device, feature, model, TNR, train_time))
        db_connection.commit()
        cursor.close()
  
    @staticmethod
    def create_ml_anomaly_testing(db_connection, device, experiment, ransomware, feature, model, TPR, test_time):
        """
        Creates a new entry in the ml_dl_anomaly_testing table.
        :input: db_connection: database
        connection, device: device that made the request, experiment: experiment that is being used,
        ransomware: ransomware that is being used, feature: feature that is being used, model: model that is being used,
        TPR: TPR of the model, test_time: time it took to test the model
        """
        cursor = db_connection.cursor()
        cursor.execute("""
            INSERT INTO ML_Testing_Anomaly (device, experiment, ransomware, feature, model, TPR, test_time)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (device, experiment, ransomware, feature, model, TPR, test_time))
        db_connection.commit()
        cursor.close()


    @staticmethod
    def create_dl_anomaly(db_connection, device, feature, model, TNR, train_time, threshold, neurons):
        """
        Creates a new entry in the dl_anomaly table.
        :input: db_connection: database
        connection, device: device that made the request, feature: feature that is being used, model:
        model that is being used, TNR: TNR of the model, train_time: time it took to train the model,
        threshold: threshold of the model, neurons: neurons of the model
        """
        cursor = db_connection.cursor()
        cursor.execute("""
            INSERT INTO DL_Training_Anomaly (device, feature, model, threshhold, neurons, TNR, train_time)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (device, feature, model, threshold, neurons, TNR, train_time))
        db_connection.commit()
        cursor.close()
    
    @staticmethod
    def get_threshold(db_connection, device, feature):
        """
        Gets the threshold of the model from the database.
        :input: db_connection: database connection, device: device that made the request, feature: 
        feature that is being used 
        :return: the threshold of the model
        """
        cursor = db_connection.cursor()
        cursor.execute("""
            SELECT threshhold FROM DL_Training_Anomaly
            WHERE device = ? AND feature = ?
        """, (device, feature))
        threshold = cursor.fetchone()
        cursor.close()
        if threshold is None:
            return None
        return threshold[0]
        
    @staticmethod
    def create_dl_anomaly_testing(db_connection, device, experiment, ransomware, feature, model, TPR, test_time):
        """
        Creates a new entry in the dl_anomaly_testing table.
        :input: db_connection: database
        connection, device: device that made the request, experiment: experiment that is being used,
        ransomware: ransomware that is being used, feature: feature that is being used, model: model that is being used,
        TPR: TPR of the model, test_time: time it took to test the model, threshold: threshold of the model, neurons: neurons of the model
        """
        cursor = db_connection.cursor()
        cursor.execute("""
            INSERT INTO DL_Testing_Anomaly (device, experiment, ransomware, feature, model, TPR, test_time)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (device, experiment, ransomware, feature, model, TPR, test_time))
        db_connection.commit()
        cursor.close()
        

        