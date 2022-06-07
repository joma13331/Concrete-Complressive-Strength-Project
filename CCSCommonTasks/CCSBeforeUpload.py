import os
import logging
import pandas as pd


class CCSBeforeUpload:
    """
    :Class Name: CCSBeforeUpload
    :Description: This class is used to transform the Good Raw Files before uploading to
                  to cassandra database

    Written By: Jobin Mathew
    Interning at iNeuron Intelligence
    Version: 1.0
    """

    def __init__(self, is_training):

        if is_training:
            self.good_raw_path = "CCSDIV/ValidatedData/GoodRaw/"
            if not os.path.isdir("CCSLogFiles/training/"):
                os.mkdir("CCSLogFiles/training/")
            self.log_path = "CCSLogFiles/training/CCSBeforeUpload.txt"
            self.operation = "TRAINING"
        else:
            self.good_raw_path = "CCSDIV/PredictionData/GoodRaw/"
            if not os.path.isdir("CCSLogFiles/prediction/"):
                os.mkdir("CCSLogFiles/prediction/")
            self.log_path = "CCSLogFiles/prediction/CCSBeforeUpload.txt"
            self.operation = "PREDICTION"

        self.ccs_before_upload_logging = logging.getLogger("ccs_before_upload_log")
        self.ccs_before_upload_logging.setLevel(logging.INFO)
        ccs_before_upload_handler = logging.FileHandler(self.log_path)
        formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s',
                                      datefmt='%m/%d/%Y %I:%M:%S %p')
        ccs_before_upload_handler.setFormatter(formatter)
        self.ccs_before_upload_logging.addHandler(ccs_before_upload_handler)

    def ee_replace_missing_with_null(self):
        """
        :Method Name: ee_replace_missing_with_null
        :Description: This method replaces all the missing values with 'null'.
        :return: None
        :On Failure: Exception
        """

        try:

            # Find all the files in the acceptable files folder and fill 'null' wherever there are missing values.
            # 'null' is being used so that cassandra database can accept missing values even in numerical columns.

            for filename in os.listdir(self.good_raw_path):
                print(filename)
                temp_df = pd.read_excel(os.path.join(self.good_raw_path, filename))
                temp_df.fillna('null', inplace=True)
                temp_df.to_excel(os.path.join(self.good_raw_path, filename), header=True, index=None)
                message = f"{self.operation}: {filename} transformed successfully"
                self.ccs_before_upload_logging.info(message)

        except Exception as e:
            message = f"Data Transformation Failed: {str(e)}"
            self.ccs_before_upload_logging.error(message)
            raise e
