import os
import pandas as pd
import logging


class CCSDataLoader:
    """
    :Class Name: CCSDataLoader
    :Description: This class contains the method for loading the data into
                  a pandas dataframe for future usage

    Written By: Jobin Mathew
    Interning at iNeuron Intelligence
    Version: 1.0
    """

    def __init__(self, is_training):
        if is_training:
            self.operation = 'TRAINING'
            self.data_file = 'validated_file.csv'
            if not os.path.isdir("CCSLogFiles/training/"):
                os.mkdir("CCSLogFiles/training/")
            self.log_path = "CCSLogFiles/training/CCSDataLoader.txt"
        else:
            self.operation = 'PREDICTION'
            self.data_file = 'prediction_file.csv'
            if not os.path.isdir("CCSLogFiles/prediction/"):
                os.mkdir("CCSLogFiles/prediction/")
            self.log_path = "CCSLogFiles/prediction/CCSDataLoader.txt"

        self.ccs_dataloader_logging = logging.getLogger("ccs_dataloader_log")
        self.ccs_dataloader_logging.setLevel(logging.INFO)
        ccs_dataloader_handler = logging.FileHandler(self.log_path)
        formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s',
                                      datefmt='%m/%d/%Y %I:%M:%S %p')
        ccs_dataloader_handler.setFormatter(formatter)
        self.ccs_dataloader_logging.addHandler(ccs_dataloader_handler)

    def ccs_get_data(self):
        """
        Method Name: ccs_get_data
        Description: This method reads the data from source.
        Output: A pandas DataFrame.
        On Failure: Raise Exception
        """
        try:

            self.data = pd.read_csv(self.data_file)
            # To round all the values to two decimal digits as it is usually in the data files.
            self.data = self.data.round(2)
            message = f"{self.operation}: The data is loaded successfully as a pandas dataframe"
            self.ccs_dataloader_logging.info(message)
            return self.data

        except Exception as e:
            message = f"{self.operation}: Error while trying to load the data for prediction to pandas dataframe: {str(e)}"
            self.ccs_dataloader_logging.error(message)
            raise e
