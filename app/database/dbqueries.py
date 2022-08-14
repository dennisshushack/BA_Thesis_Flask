class dbqueries:

    @staticmethod
    def insert_into_post_requests(db_connection, device, category, ml_type, monitors, behavior, training_path):
        """
        Inserts the data into the post_requests table.
        :input: db_connection: database connection, description: description of the request, 
        category: category of the request, type: normal, ransoRES, ransoKERN, path: path of the data
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
    def insert_into_anomaly_detection(db_connection, device, feature, model, TNR, train_time, test_time):
        """
        Creates a new entry in the ml_dl_anomaly table.
        :input: db_connection: database connection, device: device that made the request, feature: 
        feature that is being used, model: model that is being used, TNR: TNR of the model, train_time:
        time it took to train the model
        """
        cursor = db_connection.cursor()
        cursor.execute("""
            INSERT INTO ML_Training_Anomaly (device, feature, model, TNR, train_time, test_time)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (device, feature, model, TNR, train_time, test_time))
        db_connection.commit()
        cursor.close()

    @staticmethod
    def get_foreign_key_ml(db_connection, device, feature, model):
        """
        Gets the id from the ML_Training_Anomaly table.
        :input: db_connection: database connection, device: device that made the request, feature: 
        feature that is being used, model: model that is being used
        :return: the id of the entry
        """
        cursor = db_connection.cursor()
        cursor.execute("""
            SELECT id FROM ML_Training_Anomaly
            WHERE device = ? AND feature = ? AND model = ?
        """, (device, feature, model))
        id = cursor.fetchone()
        cursor.close()
        if id is None:
            return None
        return id[0]
  
    @staticmethod
    def create_ml_anomaly_testing(db_connection, device, experiment, ransomware, feature, model, TPR, pk_id):
        """
        Creates a new entry in the ml_dl_anomaly_testing table.
        :input: db_connection: database
        connection, device: device that made the request, experiment: experiment that is being used,
        ransomware: ransomware that is being used, feature: feature that is being used, model: model that is being used,
        TPR: TPR of the model, test_time: time it took to test the model
        """
        cursor = db_connection.cursor()
        cursor.execute("""
            INSERT INTO ML_Testing_Anomaly (trainings_id, device, experiment, ransomware, feature, model, TPR)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (pk_id, device, experiment, ransomware, feature, model, TPR))
        db_connection.commit()
        cursor.close()


    @staticmethod
    def create_dl_anomaly(db_connection, device, feature, model, TNR_STD, TNR_IQR, STD_lower, STD_upper,  IQR_lower, IQR_upper, neurons, train_time, test_time):
        """
        Creates a new entry in the dl_anomaly table.
        :input: db_connection: database
        connection, device: device that made the request, feature: feature that is being used, model:
        model that is being used, TNR: TNR of the model, train_time: time it took to train the model,
        threshold: threshold of the model, neurons: neurons of the model
        """
        cursor = db_connection.cursor()
        cursor.execute("""
            INSERT INTO DL_Training_Anomaly (device, feature, model, STD_lower, STD_upper, IQR_lower, IQR_upper, neurons, TNR_STD, TNR_IQR, train_time, test_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (device, feature, model, STD_lower, STD_upper, IQR_lower, IQR_upper, neurons, TNR_STD, TNR_IQR, train_time, test_time))
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
            SELECT STD_lower, STD_upper, IQR_lower, IQR_upper FROM DL_Training_Anomaly
            WHERE device = ? AND feature = ?
        """, (device, feature))
        output = cursor.fetchall()
        print(output)
        cursor.close()
        return output

    @staticmethod
    def get_foreign_key_dl(db_connection, device, feature, model):
        """
        Gets the id from the DL_Training_Anomaly table.
        :input: db_connection: database connection, device: device that made the request, feature: 
        feature that is being used, model: model that is being used
        :return: the id of the entry
        """
        cursor = db_connection.cursor()
        cursor.execute("""
            SELECT id FROM DL_Training_Anomaly
            WHERE device = ? AND feature = ? AND model = ?
        """, (device, feature, model))
        id = cursor.fetchone()
        cursor.close()
        if id is None:
            return None
        return id[0]
        
    @staticmethod
    def create_dl_anomaly_testing(db_connection, device, experiment, ransomware, feature, model, TPR_STD, TPR_IQR, pk_id):
        """
        Creates a new entry in the dl_anomaly_testing table.
        :input: db_connection: database
        connection, device: device that made the request, experiment: experiment that is being used,
        ransomware: ransomware that is being used, feature: feature that is being used, model: model that is being used,
        TPR: TPR of the model, test_time: time it took to test the model, threshold: threshold of the model, neurons: neurons of the model
        """
        cursor = db_connection.cursor()
        cursor.execute("""
            INSERT INTO DL_Testing_Anomaly (trainings_id, device, experiment, ransomware, feature, model, TPR_STD, TPR_IQR)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (pk_id, device, experiment, ransomware, feature, model, TPR_STD, TPR_IQR))
        db_connection.commit()
        cursor.close()
        
    @staticmethod
    def insert_into_live(db_connection, device, ml_type, algo, feature, time_stamp, anomaly_class, testing_time):
        """
        Inserts a new entry into the live table.
        :input: db_connection: database connection, device: device that made the request, ml_type: 
        ml_type that is being used, time_stamp: time it took to test the model, anomaly_class: anomaly class of the model
        """
        cursor = db_connection.cursor()
        cursor.execute("""
            INSERT INTO live_anomaly (device, ml_type, algo, feature, time_stamp, anomaly_class, testing_time)
            VALUES (?, ?, ?, ?, ? ,?, ?)
        """, (device, ml_type, algo, feature, time_stamp, anomaly_class, testing_time))
        db_connection.commit()
        cursor.close()

    @staticmethod
    def get_live_anomaly(db_connection, device, algo, feature):
        """
        Gets the anomaly class from the live table.
        :input: db_connection: database connection, device: device that made the request, ml_type: 
        ml_type that is being used
        :return: the anomaly class of the model
        """
        cursor = db_connection.cursor()
        cursor.execute("""
            SELECT time_stamp, anomaly_class FROM live_anomaly
            WHERE device = ? AND algo = ? AND feature = ? 
            ORDER BY time_stamp
        """, (device, algo, feature))
        anomaly_class = cursor.fetchall()
        cursor.close()
        if anomaly_class is None:
            return None
        return anomaly_class


    @staticmethod
    def get_live_anomaly_detection(db_connection, device, algo, feature):
        """
        Gets the anomaly class from the live table.
        :input: db_connection: database connection, device: device that made the request, ml_type: 
        ml_type that is being used
        :return: the anomaly class of the model
        """
        cursor = db_connection.cursor()
        cursor.execute("""
            SELECT time_stamp, anomaly_class FROM live_anomaly
            WHERE device = ? AND algo = ? AND feature = ?
            ORDER BY time_stamp
        """, (device, algo, feature))
        anomaly_class = cursor.fetchall()
        cursor.close()
        if anomaly_class is None:
            return None
        return anomaly_class

    @staticmethod
    def get_devices(db_connection):
        """
        Gets all the devices from the database.
        :input: db_connection: database connection
        :return: all the devices
        """
        cursor = db_connection.cursor()
        cursor.execute("""
            SELECT DISTINCT device FROM live_anomaly
        """)
        devices = cursor.fetchall()
        cursor.close()
        if devices is None:
            return None
        return devices


    @staticmethod
    def get_features(db_connection):
        """
        Gets all the devices from the database.
        :input: db_connection: database connection
        :return: all the devices
        """
        cursor = db_connection.cursor()
        cursor.execute("""
            SELECT DISTINCT feature FROM live_anomaly
        """)
        devices = cursor.fetchall()
        cursor.close()
        if devices is None:
            return None
        return devices
     

        