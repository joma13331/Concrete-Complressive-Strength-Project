import os
import logging
import threading

from CCSCommonTasks.CCSDataFormatValidator import CCSDataFormatValidator
from CCSCommonTasks.CCSDBOperation import CCSDBOperation
from CCSCommonTasks.CCSBeforeUpload import CCSBeforeUpload


class CCSDataInjestionComplete:
    """
        :Class Name: CCSDataInjestionComplete
        :Description: This class utilized 3 Different classes
                        1. EEDataFormatTrain
                        2. EEBeforeUploadTrain
                        3. EEDBOperationTrain
                      to complete validation on the dataset names, columns, etc based on
                      the DSA with the client. It then uploads the valid files to a cassandra
                      Database. Finally it obtains a csv from the database to be used further
                      for preprocessing and training

        Written By: Jobin Mathew
        Interning at iNeuron Intelligence
        Version: 1.0
        """

    def __init__(self, is_training, data_dir="CCSUploadedFiles"):
        """
        :Method Name: __init__
        :Description: This method initializes the variables that will be used in methods of this class.

        :param is_training: Whether this class is instantiated for training.
        :param data_dir: Data directory where files are present.
        """
        self.data_format_validator = CCSDataFormatValidator(is_training=is_training, path=data_dir)
        self.db_operator = CCSDBOperation(is_training=is_training)
        self.data_transformer = CCSBeforeUpload(is_training=is_training)

        if is_training:
            self.operation = 'TRAINING'
            if not os.path.isdir("CCSLogFiles/training/"):
                os.mkdir("CCSLogFiles/training")
            self.log_path = os.path.join("CCSLogFiles/training/", "CCSDataInjestionComplete.txt")
        else:
            self.operation = 'TRAINING'
            if not os.path.isdir("CCSLogFiles/prediction/"):
                os.mkdir("CCSLogFiles/prediction")
            self.log_path = os.path.join("CCSLogFiles/prediction/", "CCSDataInjestionComplete.txt")

        self.ccs_data_injestion_logging = logging.getLogger("ccs_data_injestion_log")
        self.ccs_data_injestion_logging.setLevel(logging.INFO)
        ccs_data_injestion_handler = logging.FileHandler(self.log_path)
        formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s',
                                      datefmt='%m/%d/%Y %I:%M:%S %p')
        ccs_data_injestion_handler.setFormatter(formatter)
        self.ccs_data_injestion_logging.addHandler(ccs_data_injestion_handler)

    def ccs_data_injestion_complete(self):
        """
        :Method Name: ccs_data_injestion_complete
        :Description: This method is used to complete the entire data validation,
                      data injestion process to store the data in a database and
                      convert it for further usage in our project work

        :return: None
        :On Failure: Exception
        """
        try:
            message = f"{self.operation}: Start of Injestion and Validation"
            self.ccs_data_injestion_logging.info(message)

            length_date, length_time, dataset_col_names, dataset_col_num = self.data_format_validator.ccs_value_from_schema()
            regex = self.data_format_validator.ccs_regex_file_name()
            self.data_format_validator.ccs_validating_file_name(regex)
            self.data_format_validator.ccs_validate_column_length(dataset_col_num)
            self.data_format_validator.ccs_validate_whole_columns_as_empty()
            self.data_format_validator.ccs_move_bad_files_to_archive()

            message = f"{self.operation}: Raw Data Validation complete"
            self.ccs_data_injestion_logging.info(message)

            message = f"{self.operation}: Start of Data Transformation"
            self.ccs_data_injestion_logging.info(message)

            self.data_transformer.ee_replace_missing_with_null()

            message = f"{self.operation}: Data Transformation Complete"
            self.ccs_data_injestion_logging.info(message)

            message = f"{self.operation}: Start of upload of the Good Data to Cassandra Database"
            self.ccs_data_injestion_logging.info(message)

            # Threading used to bypass time consuming database tasks to improve web application latency.
            t1 = threading.Thread(target=self.db_operator.ccs_complete_db_pipeline,
                                  args=[dataset_col_names, self.data_format_validator])
            t1.start()
            # t1 not joined so that it runs only after training has occurred.

            self.data_format_validator.ccs_convert_direct_excel_to_csv()

            message = f"{self.operation}: End of Injestion and Validation"
            self.ccs_data_injestion_logging.info(message)

        except Exception as e:
            message = f"{self.operation}: Error During Injestion and Validation Phase{str(e)}"
            self.ccs_data_injestion_logging.error(message)
            raise e


