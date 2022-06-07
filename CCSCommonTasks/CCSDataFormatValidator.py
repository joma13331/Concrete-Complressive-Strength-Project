import os
import re
import json
import shutil
import logging
import pandas as pd
from datetime import datetime


class CCSDataFormatValidator:
    """
    :Class Name: CCSDataFormatValidator
    :Description: This class shall be used for handling all the data validation as agreed with the
                  Client.

    Written By: Jobin Mathew
    Interning at iNeuron Intelligence
    Version: 1.0
    """

    def __init__(self, is_training, path):
        """
        :Method Name: __init__
        :Description: This method is Constructor for class CCSDataFormatValidator.
                      Initializes variables for logging.
                      Sets up the path for storing Validated Data.
        :param is_training: Whether this class is instantiated for training.
        :param path: directory path where the files for training are present.
        """

        if is_training:
            if not os.path.isdir("CCSLogFiles/training/"):
                os.mkdir("CCSLogFiles/training")
            self.log_path = os.path.join("CCSLogFiles/training/", "CCSDataFormatValidator.txt")
            self.operation = "TRAINING"
            self.dir_path = path

            self.good_raw_path = "CCSDIV/ValidatedData/GoodRaw/"
            self.bad_raw_path = "CCSDIV/ValidatedData/BadRaw/"

            self.schema_path = "CCSSchemas/training_schema.json"
            self.csv_filename = "validated_file.csv"

        else:
            if not os.path.isdir("CCSLogFiles/prediction/"):
                os.mkdir("CCSLogFiles/prediction")
            self.log_path = os.path.join("CCSLogFiles/prediction/", "CCSDataFormatValidator.txt")
            self.operation = "PREDICTION"

            self.dir_path = path

            self.good_raw_path = "CCSDIV/PredictionData/GoodRaw/"
            self.bad_raw_path = "CCSDIV/PredictionData/BadRaw/"

            self.schema_path = "CCSSchemas/prediction_schema.json"
            self.csv_filename = "prediction_file.csv"

        self.ccs_data_format_validator_logging = logging.getLogger("ccs_data_format_validator_log")
        self.ccs_data_format_validator_logging.setLevel(logging.INFO)
        ccs_data_format_handler = logging.FileHandler(self.log_path)
        formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s',
                                      datefmt='%m/%d/%Y %I:%M:%S %p')
        ccs_data_format_handler.setFormatter(formatter)
        self.ccs_data_format_validator_logging.addHandler(ccs_data_format_handler)

    def ccs_value_from_schema(self):
        """
        :Method Name: ccs_value_from_schema
        :Description: This method utilizes the json file in CCSSchema from DSA to obtain
                      the expected dataset filename and dataset column details.
        :On Failure: can Raise ValueError, KeyError or Exception
        :return: 1. length of the Year that should be in filename
                 2. length of the Time that should be in filename
                 3. column names and corresponding datatype
                 4. Number of Columns expected in the dataset
        """

        try:
            with open(self.schema_path, 'r') as f:
                dic = json.load(f)

            length_date_of_file = dic["LengthOfDate"]
            length_time_of_file = dic["LengthOfTime"]
            column_names = dic["ColumnNames"]
            print(column_names)
            column_number = dic["NumberOfColumns"]

            message = f"{self.operation}: Length of year of file = {length_date_of_file}, Length of time of file " \
                      f"= {length_time_of_file}, Number of columns = {column_number}"
            self.ccs_data_format_validator_logging.info(message)

            return length_date_of_file, length_time_of_file, column_names, column_number

        except ValueError:
            message = f"{self.operation}: ValueError:Value not found inside schema_training.json"
            self.ccs_data_format_validator_logging.error(message)
            raise ValueError

        except KeyError:
            message = f"{self.operation}:KeyError:Incorrect key passed for schema_training.json"
            self.ccs_data_format_validator_logging.error(message)
            raise KeyError

        except Exception as e:
            self.ccs_data_format_validator_logging.error(f"{self.operation}: {str(e)}")
            raise e

    def ccs_regex_file_name(self):
        """
        Method Name: ee_regex_file_name
        Description: To generate the regex to compare whether the filename is
                     according to the DSA or not
        :return: Required Regex pattern
        :On Failure: None
        """
        regex = re.compile(r'Concrete_Data_[0123]\d[01]\d[12]\d{3}_[012]\d[0-5]\d[0-5]\d.xls')
        return regex

    def ccs_create_good_bad_raw_data_directory(self):
        """
        :Method Name: ccs_create_good_bad_raw_data_directory
        :Description: This method creates directories to store the Good Data and Bad Data
                      after validating the training data.
        :return: None
        On Failure: OSError, Exception
        """
        try:

            if not os.path.isdir(self.good_raw_path):
                os.makedirs(self.good_raw_path)
            if not os.path.isdir(self.bad_raw_path):
                os.makedirs(self.bad_raw_path)

            message = f"{self.operation}: Good and Bad file directory created"
            self.ccs_data_format_validator_logging.info(message)

        except OSError as e:
            message = f"{self.operation}: Error while creating directory: {str(e)}"
            self.ccs_data_format_validator_logging.error(message)
            raise OSError

        except Exception as e:
            self.ccs_data_format_validator_logging.error(f"{self.operation}: {str(e)}")
            raise e

    def ccs_delete_existing_good_data_folder(self):
        """
        :Method Name: ccs_delete_existing_good_data_folder
        :Description: This method deletes the directory made to store the Good Data
                      after loading the data in the table. Once the good files are
                      loaded in the DB,deleting the directory ensures space optimization.
        :return: None
        :On Failure: OSError, Exception
        """

        try:

            if os.path.isdir(self.good_raw_path):
                shutil.rmtree(self.good_raw_path)
                message = f"{self.operation}: GoodRaw directory deleted successfully!!!"
                self.ccs_data_format_validator_logging.info(message)

        except OSError as e:
            message = f"{self.operation}: Error while creating directory: {str(e)}"
            self.ccs_data_format_validator_logging.error(message)
            raise e
        except Exception as e:
            self.ccs_data_format_validator_logging.error(f"{self.operation}: {str(e)}")
            raise e

    def ccs_delete_existing_bad_data_folder(self):
        """
                :Method Name: ccs_delete_existing_bad_data_folder
                :Description: This method deletes the directory made to store the Bad Data
                              after moving the data in an archive folder. We archive the bad
                              files to send them back to the client for invalid data issue.
                :return: None
                :On Failure: OSError
                """
        try:

            if os.path.isdir(self.bad_raw_path):
                shutil.rmtree(self.bad_raw_path)
                message = f"{self.operation}BadRaw directory deleted successfully!!!"
                self.ccs_data_format_validator_logging.info(message)

        except OSError as e:
            message = f"{self.operation}: Error while creating directory: {str(e)}"
            self.ccs_data_format_validator_logging.error(message)
            raise e
        except Exception as e:
            self.ccs_data_format_validator_logging.error(f"{self.operation}: {str(e)}")
            raise e

    def ccs_move_bad_files_to_archive(self):
        """
        :Method Name: ccs_move_bad_files_to_archive
        Description: This method deletes the directory made to store the Bad Data
                      after moving the data in an archive folder. We archive the bad
                      files to send them back to the client for invalid data issue.
        :return: None
        : On Failure: Exception
        """
        now = datetime.now()
        date = now.date()
        time = now.strftime("%H_%M_%S")

        try:

            if os.path.isdir(self.bad_raw_path):
                archive_dir = "CCSDIV/ArchivedData"
                if not os.path.isdir(archive_dir):
                    os.makedirs(archive_dir)
                archive_path = os.path.join(archive_dir, f"BadData_{str(date)}_{time}.xls")
                bad_files = os.listdir(self.bad_raw_path)
                for file in bad_files:
                    if file not in os.listdir(archive_dir):
                        shutil.move(self.bad_raw_path+file, archive_dir)
                        os.rename(os.path.join(archive_dir, file), archive_path)

                message = f"{self.operation}: Bad files moved to archive: {archive_path}"
                self.ccs_data_format_validator_logging.info(message)
                self.ccs_delete_existing_bad_data_folder()
        except Exception as e:
            message = f"{self.operation}: Error while Archiving Bad Files: {str(e)}"
            self.ccs_data_format_validator_logging.error(message)
            raise e

    def ccs_validating_file_name(self, regex):
        """
        :Method Name: ccs_validating_file_name
        :Description: This function validates the name of the training xlsx files as per given name in the EESchema!
                      Regex pattern is used to do the validation.If name format do not match the file is moved
                      to Bad Raw Data folder else in Good raw data.
        :param regex: The regex compiler used to check validity of filenames
        :return: None
        :On Failure: Exception
        """

        # delete the directories for good and bad data in case last run was unsuccessful and folders were not deleted.
        self.ccs_delete_existing_bad_data_folder()
        self.ccs_delete_existing_good_data_folder()
        # create new directories
        self.ccs_create_good_bad_raw_data_directory()

        raw_files = [file for file in os.listdir(self.dir_path)]

        print(raw_files)
        try:
            for filename in raw_files:
                if re.match(regex, filename):
                    shutil.copy(os.path.join(self.dir_path, filename), self.good_raw_path)
                    message = f"{self.operation}: {filename} is valid!! moved to GoodRaw folder"
                    self.ccs_data_format_validator_logging.info(message)
                else:
                    shutil.copy(os.path.join(self.dir_path, filename), self.bad_raw_path)
                    message = f"{self.operation}: {filename} is not valid!! moved to BadRaw folder"
                    self.ccs_data_format_validator_logging.info(message)

        except Exception as e:
            message = f"{self.operation}: Error occurred while validating filename: {str(e)}"
            self.ccs_data_format_validator_logging.error(message)
            raise e

    def ccs_validate_column_length(self, number_of_columns):
        """
        :Method Name: ccs_validate_column_length
        :Description: This function validates the number of columns in the csv files.
                       It is should be same as given in the EESchema file.
                       If not same file is not suitable for processing and thus is moved to Bad Raw Data folder.
                       If the column number matches, file is kept in Good Raw Data for processing.

        :param number_of_columns: The number of columns that is expected based on DSA
        :return: None
        :On Failure: OSERROR, EXCEPTION
        """
        try:

            message = f"{self.operation}: Column Length Validation Started!!"
            self.ccs_data_format_validator_logging.info(message)

            for filename in os.listdir(self.good_raw_path):
                pd_df = pd.read_excel(os.path.join(self.good_raw_path, filename))

                # Accessing the number of columns in the relevant files by checking shape of the dataframe.
                if not pd_df.shape[1] == number_of_columns:
                    shutil.move(os.path.join(self.good_raw_path, filename), self.bad_raw_path)
                    message = f"{self.operation}: invalid Column length for the file {filename}.File moved to Bad Folder"
                    self.ccs_data_format_validator_logging.info(message)
                else:
                    message = f"{self.operation}: {filename} validated. File remains in Good Folder"
                    self.ccs_data_format_validator_logging.info(message)

            message = f"{self.operation}: Column Length Validation Completed!!"
            self.ccs_data_format_validator_logging.info(message)

        except OSError:
            message = f"{self.operation}: Error occurred when moving the file: {str(OSError)}"
            self.ccs_data_format_validator_logging.error(message)
            raise OSError
        except Exception as e:
            message = f"{self.operation}: Error occurred : {str(e)}"
            self.ccs_data_format_validator_logging.error(message)
            raise e

    def ccs_validate_whole_columns_as_empty(self):
        """
        :Method Name: ccs_validate_whole_columns_as_empty
        :Description: This method validates that there are no columns in the given file
                      that has no values.
        :return: None
        :On Failure: OSError, Exception
        """
        try:

            message = f"{self.operation}: Check for Whole Columns as Empty Validation Started!!"
            self.ccs_data_format_validator_logging.info(message)
            for filename in os.listdir(self.good_raw_path):
                pd_df = pd.read_excel(os.path.join(self.good_raw_path, filename))
                for column in pd_df:
                    if (len(pd_df[column]) - pd_df[column].count()) == len(pd_df[column]):
                        shutil.move(os.path.join(self.good_raw_path, filename), self.bad_raw_path)
                        message = f"{self.operation}: invalid column {column}. Moving to Bad Folder"
                        self.ccs_data_format_validator_logging.info(message)
                        break
        except OSError:
            message = f"{self.operation}: Error occurred when moving the file: {str(OSError)}"
            self.ccs_data_format_validator_logging.error(message)
            raise OSError
        except Exception as e:
            message = f"{self.operation}: Error occurred : {str(e)}"
            self.ccs_data_format_validator_logging.error(message)
            raise e

    def ccs_convert_direct_excel_to_csv(self):
        """
        :Method Name: ccs_convert_direct_excel_to_csv
        :Description: This function converts all the excel files which have been validated as being in the correct
                      format into a single csv file which is then used in preprocessing for training ML EEModels.
                      This function is used to improve the speed or latency of the web application as the app does not
                      have to wait for database operations before starting the training.
        :return: None
        :On Failure: Exception
        """
        try:

            list_pd = []
            for filename in os.listdir(self.good_raw_path):
                list_pd.append(pd.read_excel(os.path.join(self.good_raw_path, filename)))

            df = pd.concat(list_pd)
            df.to_csv(self.csv_filename, header=True, index=True, index_label="id")

            message = f"{self.operation}: Excel file Converted directly to required csv file for future preprocessing"
            self.ccs_data_format_validator_logging.info(message)

        except Exception as e:
            message = f"{self.operation}: Error occurred while direct conversion from excel to csv: {str(e)}"
            self.ccs_data_format_validator_logging.info(message)
            raise e
