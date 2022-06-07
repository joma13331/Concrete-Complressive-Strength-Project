import os
import cassandra
from cassandra.query import dict_factory
import pandas as pd

import logging
import csv
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider


class CCSDBOperation:
    """
    This class will handle all the relevant operations related to Cassandra Database.

    Written By: Jobin Mathew
    Interning at iNeuron Intelligence
    Version: 1.0
    """

    def __init__(self, is_training):
        """
        :Method Name: __init__
        :Description: This constructor initializes the variable that will be utilized
                      in all the class methods
        :param is_training: Boolean variable to inform whether training has to be done
        """

        if is_training:
            if not os.path.isdir("CCSLogFiles/training/"):
                os.mkdir("CCSLogFiles/training/")

            self.operation = "TRAINING"
            self.log_path = os.path.join("CCSLogFiles/training", "CCSDBOperation.txt")

            self.good_file_dir = "CCSDIV/ValidatedData/GoodRaw/"
            self.bad_file_dir = "CCSDIV/ValidatedData/BadRaw/"
            self.table_name = "good_training_data"
        else:
            if not os.path.isdir("CCSLogFiles/prediction/"):
                os.mkdir("CCSLogFiles/prediction/")

            self.operation = "PREDICTION"
            self.log_path = os.path.join("CCSLogFiles/prediction/", "CCSDBOperation.txt")

            self.good_file_dir = "CCSDIV/PredictionData/GoodRaw/"
            self.bad_file_dir = "CCSDIV/PredictionData/BadRaw/"
            self.table_name = "good_prediction_data"

        self.ccs_db_operation_logging = logging.getLogger("ccs_db_operation_log")
        self.ccs_db_operation_logging.setLevel(logging.INFO)
        ccs_db_operation_handler = logging.FileHandler(self.log_path)
        formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s',
                                      datefmt='%m/%d/%Y %I:%M:%S %p')
        ccs_db_operation_handler.setFormatter(formatter)
        self.ccs_db_operation_logging.addHandler(ccs_db_operation_handler)

    def ccs_db_connection(self):
        """
        :Method Name: ccs_db_connection
        :Description: This method connects to the keyspace used for storing the validated
                      good dataset for this work.
        :return: session which is a cassandra database connection
        :On Failure: cassandra.cluster.NoHostAvailable, Exception
        """
        try:

            cloud_config = {
                'secure_connect_bundle': 'secure-connect-ineuron.zip'
            }
            auth_provider = PlainTextAuthProvider(os.getenv('CASSANDRA_CLIENT_ID'),
                                                  os.getenv('CASSANDRA_CLIENT_SECRET'))
            cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)

            session = cluster.connect()
            session.row_factory = dict_factory
            message = f"{self.operation}: Connection successful with cassandra database"
            self.ccs_db_operation_logging.info(message)

            session.execute("USE concrete_compressive_strength_internship;")

            message = f"{self.operation}: accessed the concrete_compressive_strength_internship keyspace"
            self.ccs_db_operation_logging.info(message)

            return session

        except cassandra.cluster.NoHostAvailable:
            message = f"{self.operation}: Connection Unsuccessful with cassandra database due to Incorrect " \
                      f"credentials or no connection from datastax"
            self.ccs_db_operation_logging.error(message)
            raise cassandra.cluster.NoHostAvailable
        except Exception as e:
            message = f"{self.operation}: Connection Unsuccessful with cassandra database: {str(e)}"
            self.ccs_db_operation_logging.error(message)
            raise e

    def ccs_create_table(self, column_names):
        """
        :Method Name: ccs_create_table
        :Description: This method creates a 'good_training_data' or 'good_prediction_data' table to store good data
                      with the appropriate column names.
        :param column_names: Column Names as expected from EESchema based on DSA
        :return:None
        :On Failure: Exception
        """

        try:

            session = self.ccs_db_connection()

            table_creation_query = f"CREATE TABLE IF NOT EXISTS {self.table_name}(id int primary key,"
            for col_name in column_names:
                table_creation_query += f"\"{col_name}\" {column_names[col_name]},"
            # table_creation_query[:-1] is used to not consider the ',' at the end.
            table_creation_query = table_creation_query[:-1] + ");"
            print(table_creation_query)
            session.execute(table_creation_query)
            message = f"{self.operation}: The table for Good Data created"
            self.ccs_db_operation_logging.info(message)

            session.execute(f"truncate table {self.table_name};")
            message = f"{self.operation}: Any row if existing deleted"
            self.ccs_db_operation_logging.info(message)

        except Exception as e:
            message = f"{self.operation}: The table for Good Data was Not created: {str(e)}"
            self.ccs_db_operation_logging.info(message)
            raise e

        finally:
            try:
                session.shutdown()
                message = f"{self.operation}: Session terminated create table operation"
                self.ccs_db_operation_logging.info(message)

            except Exception as e:
                pass

    def ccs_insert_good_data(self):
        """
        :Method Name: ccs_insert_good_data
        :Description: This method uploads all the files in the good Data folder
                      to the good_data tables in cassandra database.
        :return: None
        :On Failure: Exception
        """
        try:

            count = 0
            col_names = "id,"
            session = self.ccs_db_connection()

            for filename in os.listdir(self.good_file_dir):
                temp_df = pd.read_excel(os.path.join(self.good_file_dir, filename))

                # count variable is used so the the column part of the query is created only once as it is same for all
                # the insertion queries
                if count == 0:
                    for i in list(temp_df.columns):
                        col_names += f"\"{str(i).rstrip()}\","
                    # col_names[:-1] is used to not consider the ',' at the end
                    col_names = col_names[:-1]
                    count += 1

                    print(col_names)
                # the for loop creates the values to be uploaded.
                # it is complicated to ensure that any 'null' value in a string column is entered as null and not a
                # simple string.
                for i in range(len(temp_df)):
                    # [i] is the value for id.

                    temp_lis = [i] + list(temp_df.iloc[i])
                    if 'null' in temp_lis:
                        tup = "("
                        for j in temp_lis:
                            if type(j) == str:
                                if j == 'null':
                                    tup += f"{j},"
                                else:
                                    tup += f"'{j}',"
                            else:
                                tup += f"{j},"
                        tup = tup[:-1] + ")"
                    else:
                        tup = tuple(temp_lis)
                    insert_query = f"INSERT INTO {self.table_name}({col_names}) VALUES {tup};"
                    print(insert_query)
                    session.execute(insert_query)

                message = f"{self.operation}: Data in {filename} uploaded successfully to good_data table"
                self.ccs_db_operation_logging.info(message)

        except Exception as e:
            message = f"{self.operation}: Error while uploading data to good_data table: {str(e)}"
            self.ccs_db_operation_logging.error(message)
            raise e

        finally:
            try:
                session.shutdown()
                message = f"{self.operation}: Session terminated insert data operation"
                self.ccs_db_operation_logging.info(message)

            except Exception as e:
                pass

    def ccs_data_from_db_to_csv(self):
        """
        :Method Name: ccs_data_from_db_to_csv
        :Description: This method downloads all the good data from the cassandra
                      database to a csv file for preprocessing and training.
        :return: None
        :On Failure: Exception
        """
        try:
            session = self.ccs_db_connection()
            if self.operation == 'TRAINING':
                data_file = 'validated_file.csv'
                col_name_query = "select column_name from system_schema.columns where keyspace_name=" \
                                 "'concrete_compressive_strength_internship' and table_name='good_training_data'; "
            else:
                data_file = 'prediction_file.csv'
                col_name_query = "select column_name from system_schema.columns where keyspace_name=" \
                                 "'concrete_compressive_strength_internship' and table_name='good_prediction_data'; "

            headers = []
            result = session.execute(col_name_query)
            for i in result:

                headers.append(str(i['column_name']))
                print(i['column_name'])

            get_all_data_query = f"select * from {self.table_name};"
            results = session.execute(get_all_data_query)
            data = []

            for result in results:
                row = []
                for header in headers:
                    # lower() because cassandra database converts all column names to lower case.
                    row.append(result[header])
                data.append(row)

            with open(data_file, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(headers)
                csv_writer.writerows(data)

            message = f"{self.operation}: All data from good data table saved to {data_file}"
            self.ccs_db_operation_logging.info(message)

        except Exception as e:
            message = f"{self.operation}: Error while downloading good data into csv file: {str(e)}"
            self.ccs_db_operation_logging.error(message)
            raise e

        finally:
            try:
                session.shutdown()
                message = f"Session terminated after downloading good data into csv file from table{self.table_name}"
                self.ccs_db_operation_logging.info(message)

            except Exception as e:
                pass

    def ccs_complete_db_pipeline(self, column_names, data_format_validator):
        """
        :Method Name: ccs_complete_db_pipeline
        :Description: This methods is written so that it can be run on a background thread to make ensure our web app
                      first makes the prediction to ensure less latency.
                      Only after the prediction is displayed on the web app does the database operations begin.
        :param column_names: The column names of the table in the cassandra database.
        :param data_format_validator: An object of EEDataFormatPred class to perform deletion and transfer of files
        :return: None
        :On Failure: Exception
        """
        try:
            self.ccs_create_table(column_names=column_names)
            self.ccs_insert_good_data()
            data_format_validator.ccs_delete_existing_good_data_folder()
            data_format_validator.ccs_move_bad_files_to_archive()
            self.ccs_data_from_db_to_csv()

        except Exception as e:
            message = f"{self.operation}: Error in Database Pipeline: {str(e)}"
            self.ccs_db_operation_logging.error(message)
            raise e
