import os
import json
import logging
import pandas as pd

from CCSCommonTasks.CCSFileOperations import CCSFileOperations
from CCSCommonTasks.CCSDataLoader import CCSDataLoader
from CCSCommonTasks.CCSEda import CCSEda
from CCSCommonTasks.CCSFeatureEngineering import CCSFeatureEngineering
from CCSCommonTasks.CCSFeatureSelection import CCSFeatureSelection


class CCSPredictionPipeline:
    """
    :Class Name: CCSPredictionPipeline
    :Description: This class contains the method that will perform the prediction of the
                  data submitted by the client

    Written By: Jobin Mathew
    Interning at iNeuron Intelligence
    Version: 1.0
    """

    def __init__(self):
        """
        :Method Name: __init__
        :Description: This constructor sets up the logging feature and paths where the models and
                      relevant information are stored
        :return: None
        """

        self.operation = 'PREDICTION'
        if not os.path.isdir("CCSLogFiles/prediction/"):
            os.mkdir("CCSLogFiles/prediction")
        self.log_path = os.path.join("CCSLogFiles/prediction/", "CCSPredictionPipeline.txt")

        self.ccs_prediction_pipeline_logging = logging.getLogger("ccs_prediction_pipeline_log")
        self.ccs_prediction_pipeline_logging.setLevel(logging.INFO)
        ccs_prediction_pipeline_handler = logging.FileHandler(self.log_path)
        formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s',
                                      datefmt='%m/%d/%Y %I:%M:%S %p')
        ccs_prediction_pipeline_handler.setFormatter(formatter)
        self.ccs_prediction_pipeline_logging.addHandler(ccs_prediction_pipeline_handler)

        if not os.path.isdir("CCSModels/CCSMLModels/"):
            os.mkdir("CCSModels/CCSMLModels/")
        self.ml_model_dir = "CCSModels/CCSMLModels/"

        if not os.path.isdir("CCSModels/"):
            os.mkdir("CCSModels/")
        self.models_dir = "CCSModels/"

        if not os.path.isdir("CCSRelInfo/"):
            os.mkdir("CCSRelInfo/")
        self.rel_info_dir = "CCSRelInfo/"

    def ccs_predict(self):
        """
        :Method Name: ccs_predict
        :Description: This method implements the prediction pipeline which will predict on
                      the client data during deployment.
        :return: the features and their corresponding predicted labels as a json object in string format
        """
        try:
            message = f"{self.operation}: Start of Prediction Pipeline"
            self.ccs_prediction_pipeline_logging.info(message)

            data_loader = CCSDataLoader(is_training=False)
            prediction_data = data_loader.ccs_get_data()

            message = f"{self.operation}: Data to predict on obtained"
            self.ccs_prediction_pipeline_logging.info(message)

            eda = CCSEda(is_training=False)
            feature_engineer = CCSFeatureEngineering(is_training=False)
            feature_selector = CCSFeatureSelection(is_training=False)
            file_operator = CCSFileOperations()

            features = feature_selector.ccs_remove_columns(prediction_data, 'id')
            print(features.shape)

            is_null_present, columns_with_null = eda.ccs_features_with_missing_values(features)

            if is_null_present:
                features = feature_engineer.ccs_handling_missing_data_mcar(features, columns_with_null)

            with open(os.path.join(self.rel_info_dir, "columns_to_drop.txt")) as f:
                val = f.read()

            col_to_drop = val.split(",")
            if col_to_drop[0] == '':
                col_to_drop = []
            print(len(col_to_drop), type(col_to_drop))
            features = feature_selector.ccs_remove_columns(features, col_to_drop)
            # print(features.shape)
            features = feature_engineer.ccs_standard_scaling_features(features)

            cluster = file_operator.ccs_load_model(os.path.join(self.models_dir, "cluster.pickle"))

            features['clusters'] = cluster.predict(features)
            features['id'] = prediction_data['id']

            result = []

            for i in features["clusters"].unique():
                cluster_data = features[features["clusters"] == i]
                id1 = cluster_data['id']
                cluster_data = cluster_data.drop(columns=["clusters", 'id'])
                model = file_operator.ccs_load_ml_model(i)
                pred_result = list(model.predict(cluster_data))
                result.extend(list(zip(id1, pred_result)))

            res_dataframe = pd.DataFrame(data=result, columns=["id", 'Concrete compressive strength(MPa, megapascals)'])
            prediction_data = prediction_data.merge(right=res_dataframe, on='id', how='outer')

            prediction_data = prediction_data.round(2)
            prediction_data = prediction_data.drop(columns=["id"])
            prediction_data.to_csv("prediction_result.csv", header=True, index=False)

            message = f"{self.operation}: End of EEPrediction Pipeline"
            self.ccs_prediction_pipeline_logging.info(message)

            return json.loads(prediction_data.to_json(orient="records"))

        except Exception as e:
            message = f"{self.operation}: There was an ERROR while performing prediction on given data: {str(e)}"
            self.ccs_prediction_pipeline_logging.error(message)
            raise e

