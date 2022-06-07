import os
import pandas as pd
import logging
from sklearn.feature_selection import mutual_info_regression


class CCSFeatureSelection:
    """
    :Class Name: ccsFeatureSelectionTrain
    :Description: This class is used to select the features for both training as well make the same selection
                  for training.

    Written By: Jobin Mathew
    Interning at iNeuron Intelligence
    Version: 1.0
    """

    def __init__(self, is_training):
        """
        :Method Name: __init__
        :Description: This method sets up the path variables and initializes variables for logging.
        """
        if is_training:
            self.operation = 'TRAINING'
        else:
            self.operation = 'PREDICTION'

        if not os.path.isdir("CCSLogFiles/"):
            os.mkdir("CCSLogFiles/")
        self.log_path = os.path.join("CCSLogFiles/", "CCSFeatureSelection.txt")

        self.ccs_feature_selection_logging = logging.getLogger("ccs_feature_selection_log")
        self.ccs_feature_selection_logging.setLevel(logging.INFO)
        ccs_feature_selection_handler = logging.FileHandler(self.log_path)
        formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s',
                                      datefmt='%m/%d/%Y %I:%M:%S %p')
        ccs_feature_selection_handler.setFormatter(formatter)
        self.ccs_feature_selection_logging.addHandler(ccs_feature_selection_handler)

    def ccs_remove_columns(self, dataframe, columns):
        """
        :Method Name: ccs_remove_columns
        :Description: This method is used to delete columns from a pandas dataframe.

        :param dataframe: The pandas dataframe from which the columns have to be
                          removed.
        :param columns: The columns that have to be removed.
        :return: A pandas dataframe with the columns removed.
        :On Failure: Exception
        """
        try:
            dataframe = dataframe.drop(columns=columns)
            message = f"{self.operation}: The following columns were dropped: {columns}"
            self.ccs_feature_selection_logging.info(message)
            return dataframe

        except Exception as e:
            message = f"{self.operation}: There was an ERROR while deleting columns: {str(e)}"
            self.ccs_feature_selection_logging.error(message)
            raise e

    def ccs_col_with_high_correlation(self, dataframe, threshold=0.8):
        """
        :Method Name: ccs_col_with_high_correlation
        :Description: This method finds out those features which can be removed to remove multi-collinearity.

        :param dataframe: The pandas dataframe to check for features with multi-collinearity
        :param threshold: The threshold above which features are taken to be collinear
        :return: A list of features that can be dropped to remove multi-collinearity
        :On Failure: Exception
        """
        try:
            col_corr = set()  # Set of all the names of correlated columns
            corr_matrix = dataframe.corr()
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if abs(corr_matrix.iloc[i, j]) > threshold:  # we are interested in absolute coeff value
                        colname = corr_matrix.columns[i]  # getting the name of column
                        col_corr.add(colname)

            message = f"{self.operation}: The following columns have high correlation with other " \
                      f"columns {str(col_corr)}"
            self.ccs_feature_selection_logging.info(message)
            return list(col_corr)

        except Exception as e:
            message = f"There was an ERROR while detecting collinear columns in features: {str(e)}"
            self.ccs_feature_selection_logging.error(message)
            raise e

    def ccs_feature_not_important(self, features, label, threshold=0.1):
        """
        :Method Name: ccs_feature_not_important
        :Description: This method determined those features which are not important to determine the output label

        :param features: The input features of the dataset provided by the client
        :param label: The output label being considered for determining feature to drop
        :param threshold: the value below which if columns have value they can be removed
        :return: A list of features that can be dropped as they have no impact on output label
        :On Failure: Exception
        """
        try:
            mutual_info = mutual_info_regression(features, label)
            feature_imp = pd.Series(mutual_info, index=features.columns)
            not_imp = list(feature_imp[feature_imp < threshold].index)

            message = f"{self.operation}: The features which have no or very impact on the output are {not_imp}"
            self.ccs_feature_selection_logging.info(message)

            return not_imp

        except Exception as e:
            message = f"{self.operation}: There was an ERROR while detecting columns in features with no impact on " \
                      f"output: {str(e)} "
            self.ccs_feature_selection_logging.error(message)
            raise e

    def ee_features_with_zero_std(self, dataframe):
        """
        :Method Name: ee_features_with_zero_std
        :Description: This method checks whether any of the columns of the passed
                      dataframe has all values as equal and returns a list of all such
                      columns

        :param dataframe: The pandas dataframe to check for columns with all values as same
        :return: list of columns with zero std
        :On Failure: Exception
        """
        try:
            columns_zero_std = []

            for feature in dataframe.columns:
                if dataframe[feature].std() == 0:
                    columns_zero_std.append(feature)

            message = f"{self.operation}: the features with all values as equal are {columns_zero_std}"
            self.ccs_feature_selection_logging.info(message)

            return columns_zero_std

        except Exception as e:
            message = f"{self.operation}: There was an ERROR while detecting columns all values as equal: {str(e)}"
            self.ccs_feature_selection_logging.error(message)
            raise e
