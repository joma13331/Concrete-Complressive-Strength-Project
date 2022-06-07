import os
import json
import logging
import numpy as np
from scipy.stats import normaltest, boxcox


class CCSEda:
    """
    :Class Name: CCSEda
    :Description: This class is used to explore the data given by the client and come
                  to some conclusion about the data.

    Written By: Jobin Mathew
    Interning at iNeuron Intelligence
    Version: 1.0
    """

    def __init__(self, is_training):
        """
        :Method Name: __init__
        :Description: This method is Constructor for class CCSEda.
                      Initializes variables for logging.
        :param is_training: whether this class is instantiated for training purpose.
        """

        if is_training:
            if not os.path.isdir("CCSLogFiles/training"):
                os.mkdir("CCSLogFiles/training")
            self.log_path = os.path.join("CCSLogFiles/training", "CCSEda.txt")
            self.operation = "TRAINING"
        else:
            if not os.path.isdir("CCSLogFiles/prediction"):
                os.mkdir("CCSLogFiles/prediction")
            self.log_path = os.path.join("CCSLogFiles/prediction", "CCSEda.txt")
            self.operation = "PREDICTION"
        self.cont_feat_names_path = 'CCSRelInfo/Continuous_Features.txt'
        self.discrete_feat_names_path = 'CCSRelInfo/Discrete_Features.txt'
        self.normal_feature_path = 'CCSRelInfo/Normal_Features.txt'
        self.boxcox_transformed_feature = 'CCSRelInfo/BoxCox_Features.txt'
        self.log_transformed_feature = 'CCSRelInfo/Log_Features.txt'

        self.ccs_eda_logging = logging.getLogger("ccs_eda_log")
        self.ccs_eda_logging.setLevel(logging.INFO)
        ccs_eda_handler = logging.FileHandler(self.log_path)
        formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s',
                                      datefmt='%m/%d/%Y %I:%M:%S %p')
        ccs_eda_handler.setFormatter(formatter)
        self.ccs_eda_logging.addHandler(ccs_eda_handler)

    def ccs_feature_label_split(self, dataframe, label_col_names):
        """
        :Method Name: ccs_feature_label_split
        :Description: This method splits the features and labels from the validated
                      dataset and it returns them

        :param dataframe: The pandas dataframe to obtain features and labels from
        :param label_col_names: the name of label columns
        :return: features - a pandas dataframe composed of all the features
                 labels - a dataseries representing the output
        :On Failure: Exception
        """

        try:
            features = dataframe.drop(columns=label_col_names)
            labels = dataframe[label_col_names]

            message = f"{self.operation}: The features and labels have been obtained from the dataset"
            self.ccs_eda_logging.info(message)

            return features, labels

        except Exception as e:
            message = f"{self.operation}: There was an ERROR while obtaining feature and labels: {str(e)}"
            self.ccs_eda_logging.error(message)
            raise e

    def ccs_features_with_missing_values(self, dataframe):
        """
        :Method Name: ccs_features_with_missing_values
        :Description: This method finds out whether there are missing values in the
                      validated data and returns a list of feature names with missing
                      values

        :param dataframe: the Dataframe in which features with missing values are
                          required to be found
        :return: missing_val_flag - whether the dataframe has missing values or not
                 features_with_missing - If missing values are present then list of
                 columns with missing values otherwise an empty list
        :On Failure: Exception
        """
        try:
            features_with_missing = [feature for feature in dataframe.columns if dataframe[feature].isna().sum() > 0]
            missing_val_flag = False
            if len(features_with_missing) > 0:
                missing_val_flag = True

            message = f"{self.operation}: There are {len(features_with_missing)} features with missing values"
            self.ccs_eda_logging.info(message)

            return missing_val_flag, features_with_missing

        except Exception as e:
            message = f"{self.operation}: There was an ERROR while obtaining feature and labels: {str(e)}"
            self.ccs_eda_logging.error(message)
            raise e

    def ccs_continuous_discrete_variables(self, dataframe, num_col):
        """
        :Method Name: ccs_continuous_discrete_variables
        :param dataframe: The dataframe from which the column name of continuous and discrete features have to be
                          obtained
        :param num_col: List of all the numerical columns in the dataframe
        :return: cont_feat - list of continuous features in the given dataframe
                 discrete_feat - list of discrete features in the given dataframe

        """
        try:
            if self.operation == 'TRAINING':

                cont_feat = [feature for feature in num_col if len(dataframe[feature].unique()) >= 25]
                cont_feat_str = ",".join(cont_feat)
                with open(self.cont_feat_names_path, 'w') as f:
                    f.write(cont_feat_str)

                message = f'{self.operation}: {cont_feat_str} are the continuous features'
                self.ccs_eda_logging.info(message)

                discrete_feat = [feature for feature in num_col if len(dataframe[feature].unique()) < 25]
                discrete_feat_str = ",".join(discrete_feat)
                with open(self.discrete_feat_names_path, 'w') as f:
                    f.write(discrete_feat_str)

                message = f'{self.operation}: {discrete_feat_str} are the Discrete features'
                self.ccs_eda_logging.info(message)

            else:

                with open(self.cont_feat_names_path, 'r') as f:
                    cont_feat_str = f.read()
                cont_feat = cont_feat_str.split(',')
                message = f'{self.operation}: {cont_feat_str} are the continuous features'
                self.ccs_eda_logging.info(message)

                with open(self.discrete_feat_names_path, 'r') as f:
                    discrete_feat_str = f.read()
                discrete_feat = discrete_feat_str.split(',')
                message = f'{self.operation}: {discrete_feat_str} are the Discrete features'
                self.ccs_eda_logging.info(message)

            return cont_feat, discrete_feat

        except Exception as e:
            message = f'{self.operation}: Error in obtaining the Continuous and Discrete ' \
                      f'features from the data: {str(e)}'
            self.ccs_eda_logging.error(message)

    def ccs_normal_not_normal_distributed_features(self, dataframe, cont_columns):
        """
        :Method Name: ccs_normal_not_normal_distributed_features
        :param dataframe: the dataframe which needs to be checked for normal and not normal features.
        :param cont_columns: the list of continuous columns
        :return: normal_features - list of normal features
                 not_normal_features - list of features which are not normal
        """
        try:

            normal_features = []
            not_normal_features = []
            for feature in cont_columns:
                if normaltest(dataframe[feature].values)[1] >= 0.05:
                    normal_features.append(feature)
                else:
                    not_normal_features.append(feature)
            message = f'{self.operation}: {normal_features} are originally normal'
            self.ccs_eda_logging.info(message)

            normal_features_str = ','.join(normal_features)
            with open(self.normal_feature_path, 'w') as f:
                f.write(normal_features_str)

            return normal_features, not_normal_features

        except Exception as e:
            message = f"{self.operation}: There was an ERROR while obtaining normal " \
                      f"features and features which are not normal: {str(e)}"
            self.ccs_eda_logging.error(message)
            raise e

    def ccs_obtain_normal_features(self, dataframe, cont_columns):
        """
        :Method Name: ccs_obtain_normal_features
        :param dataframe: The dataframe which needs to convert its columns to normal if possible.
        :param cont_columns: the features which are continuous in nature
        :return:
        """
        try:
            if self.operation == 'TRAINING':

                normal_features, not_normal_features = self.ccs_normal_not_normal_distributed_features(dataframe, cont_columns)
                box_cox_lambda_values = {}
                log_normal_features = []
                for feature in not_normal_features:
                    data = dataframe[feature].copy(deep=True)
                    pos_data = data[data > 0]
                    boxcox_values, box_cox_lambda = boxcox(pos_data)
                    data[data > 0] = boxcox_values
                    data[data == 0] = -1 / box_cox_lambda

                    if normaltest(data)[1] >= 0.05:
                        normal_features.append(feature)
                        not_normal_features.remove(feature)
                        box_cox_lambda_values[feature] = box_cox_lambda
                        dataframe[feature] = data
                        continue

                    if normaltest(np.log(1 + dataframe[feature]))[1] > 0.05:
                        normal_features.append(feature)
                        not_normal_features.remove(feature)
                        log_normal_features.append(feature)
                        dataframe[feature] = np.log(1 + dataframe[feature])

                box_cox_strings = json.dumps(box_cox_lambda_values)
                with open(self.boxcox_transformed_feature, 'w') as f:
                    f.write(box_cox_strings)

                log_feature_string = ','.join(log_normal_features)
                with open(self.log_transformed_feature,'w') as f:
                    f.write(log_feature_string)

            else:

                with open(self.boxcox_transformed_feature, 'r') as f:
                    box_cox_strings = f.read()
                    box_cox_lambda_values = json.loads(box_cox_strings)

                with open(self.log_transformed_feature, 'r') as f:
                    log_feature_string = f.read()
                    log_normal_features = log_feature_string.split(',')

                for feature in box_cox_lambda_values:
                    data = dataframe[feature].copy(deep=True)
                    pos_data = data[data > 0]
                    data[data > 0] = boxcox(pos_data, lmbda=box_cox_lambda_values[feature])
                    data[data == 0] = -1 / box_cox_lambda_values[feature]
                    dataframe[feature] = data

                for feature in log_normal_features:
                    dataframe[feature] = np.log(1 + dataframe[feature])

            return dataframe

        except Exception as e:
            message = f"{self.operation}: There was an ERROR while converting all possible columns to normal " \
                      f"features : {str(e)}"
            self.ccs_eda_logging.error(message)
            raise e
