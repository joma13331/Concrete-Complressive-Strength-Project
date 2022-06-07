import logging
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from CCSCommonTasks.CCSFileOperations import CCSFileOperations
from sklearn.preprocessing import StandardScaler


class CCSFeatureEngineering:
    """
    :Class Name: EEFeatureEngineeringPred
    :Description: This class is used to modify the dataframe while performing data
                  preprocessing

    Written By: Jobin Mathew
    Interning at iNeuron Intelligence
    Version: 1.0
    """

    def __init__(self, is_training):
        """
        :Method Name: __init__
        :Description: it initializes the logging and various variables used in the class.

        :param is_training: Whether this class has been instantiated
        """

        if is_training:
            if not os.path.isdir("CCSLogFiles/training/"):
                os.mkdir("CCSLogFiles/training")
            self.log_path = os.path.join("CCSLogFiles/training/", "CCSFeatureEngineering.txt")
            self.operation = "TRAINING"
        else:
            if not os.path.isdir("CCSLogFiles/prediction/"):
                os.mkdir("CCSLogFiles/prediction")
            self.log_path = os.path.join("CCSLogFiles/prediction/", "CCSFeatureEngineering.txt")
            self.operation = "PREDICTION"
        self.scalar_path = "CCSModels/"
        self.imputer_path = "CCSModels/"
        self.file_operator = CCSFileOperations()

        self.ccs_feature_engineering_logging = logging.getLogger("ccs_feature_engineering_log")
        self.ccs_feature_engineering_logging.setLevel(logging.INFO)
        ccs_feature_engineering_handler = logging.FileHandler(self.log_path)
        formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s',
                                      datefmt='%m/%d/%Y %I:%M:%S %p')
        ccs_feature_engineering_handler.setFormatter(formatter)
        self.ccs_feature_engineering_logging.addHandler(ccs_feature_engineering_handler)

    def ccs_standard_scaling_features(self, dataframe):
        """
        :Method Name: ccs_standard_scaling_features
        :Description: This method takes in a dataframe and scales it using standard scalar
        :param dataframe: this is the dataframe that needs to be scaled
        :return: The Scaled dataset.

        :On Failure: Exception
        """
        try:

            if self.operation == 'TRAINING':
                scalar = StandardScaler()
                scaled_df = pd.DataFrame(scalar.fit_transform(dataframe), columns=dataframe.columns)
                message = "The dataset has been scaled using Standard Scalar"
                self.ccs_feature_engineering_logging.info(message)
                self.file_operator.ccs_save_model(scalar, self.scalar_path, 'scalar.pickle')
                return scaled_df
            else:
                scalar = self.file_operator.ccs_load_model(os.path.join(self.scalar_path, "scalar.pickle"))
                scaled_df = pd.DataFrame(scalar.transform(dataframe), columns=dataframe.columns)
                message = "The dataset has been scaled using Standard Scalar"
                self.ccs_feature_engineering_logging.info(message)
                return scaled_df

        except Exception as e:
            message = f"Error while trying to scale data: {str(e)}"
            self.ccs_feature_engineering_logging.info(message)
            raise e

    def ccs_handling_missing_data_mcar(self, dataframe, feature_with_missing):
        """
        :Method Name: ccs_handling_missing_data_mcar
        :Description: This method replaces the missing values if there are not greater than 75% missing using KNNImputer
        :param dataframe: The dataframe where null values have to be replaced
        :param feature_with_missing: The features where
        :return: dataframe - features with imputed values
                 dropped_features - features with more than 75% null
        """
        try:
            if self.operation == 'TRAINING':
                dropped_features = []
                imputer = KNNImputer(n_neighbors=3, weights='uniform', missing_values=np.nan)

                for feature in feature_with_missing:
                    if dataframe[feature].isna().mean > 0.75:
                        dataframe.drop(columns=feature)
                        dropped_features.append(feature)
                        message = f" Dropped {feature} as more than 75% values are missing"
                        self.ccs_feature_engineering_logging.info(message)

                    else:
                        dataframe[feature + 'nan'] = np.where(dataframe[feature].isnull(), 1, 0)

                data = imputer.fit_transform(dataframe)
                self.file_operator.ccs_save_model(imputer, self.imputer_path, "imputer.pickle")
                dataframe = pd.DataFrame(data=data, columns=dataframe.columns)

                message = f" missing values imputed using KNNImputer " \
                          f"for {list(set(feature_with_missing).symmetric_difference(set(dropped_features)))} "
                self.ccs_feature_engineering_logging.info(message)
                return dataframe, dropped_features

            else:
                if os.path.isfile(os.path.join(self.imputer_path, "imputer.pickle")):
                    imputer = self.file_operator.ccs_load_model(os.path.join(self.imputer_path, "imputer.pickle"))
                    data = imputer.transform(dataframe)
                    dataframe = pd.DataFrame(data=data, columns=dataframe.columns)
                    message = f" missing values imputed using KNNImputer " \
                              f"for {list(set(feature_with_missing))} "
                    self.ccs_feature_engineering_logging.info(message)
                return dataframe

        except Exception as e:
            message = f"Error while trying to handle missing data due to mcar: {str(e)}"
            self.ccs_feature_engineering_logging.error(message)
            raise e


