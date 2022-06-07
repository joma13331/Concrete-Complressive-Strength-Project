import os
import logging

from sklearn.model_selection import train_test_split

from CCSCommonTasks.CCSFileOperations import CCSFileOperations
from CCSCommonTasks.CCSDataLoader import CCSDataLoader
from CCSCommonTasks.CCSEda import CCSEda
from CCSCommonTasks.CCSFeatureEngineering import CCSFeatureEngineering
from CCSCommonTasks.CCSFeatureSelection import CCSFeatureSelection
from CCSTraining.CCSClusteringTrain import CCSClusteringTrain
from CCSTraining.CCSModelFinderTrain import CCSModelFinderTrain


class CCSTrainingPipeline:
    """
    :Class Name: CCSTrainingPipeline
    :Description: This class contains the methods which integrates all the relevant classes and their methods
                  to perform data preprocessing, training and saving of the best model for later predictions.

    Written By: Jobin Mathew
    Interning at iNeuron Intelligence
    Version: 1.0
    """

    def __init__(self):
        """
        :Method Name: __init__
        :Description: This constructor sets up the path variables where logs and models will be stored.
                      Sets up logging.
        """
        self.operation = 'TRAINING'

        if not os.path.isdir("CCSLogFiles/training/"):
            os.mkdir("CCSLogFiles/training")
        self.log_path = os.path.join("CCSLogFiles/training/", "CCSTrainingPipeline.txt")

        self.ccs_training_pipeline_logging = logging.getLogger("ccs_training_pipeline_log")
        self.ccs_training_pipeline_logging.setLevel(logging.INFO)
        ccs_training_pipeline_handler = logging.FileHandler(self.log_path)
        formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s',
                                      datefmt='%m/%d/%Y %I:%M:%S %p')
        ccs_training_pipeline_handler.setFormatter(formatter)
        self.ccs_training_pipeline_logging.addHandler(ccs_training_pipeline_handler)

        if not os.path.isdir("CCSModels/CCSMLModels/"):
            os.mkdir("CCSModels/CCSMLModels/")
        self.ml_model_dir = "CCSModels/CCSMLModels/"

        if not os.path.isdir("CCSModels/"):
            os.mkdir("CCSModels/")
        self.cluster_dir = "CCSModels/"

        if not os.path.isdir("CCSRelInfo/"):
            os.mkdir("CCSRelInfo/")
        self.rel_info_dir = "CCSRelInfo/"

    def ccs_model_train(self):
        """
        :Method Name: ccs_model_train
        :Description: This method integrates all the relevant classes and their methods to perform
                      Data Preprocessing, Clustering and saving the best model for each of the cluster.
        :return: None
        :On Failure: Exception
        """

        try:
            message = f"{self.operation}: Start of Training Pipeline"
            self.ccs_training_pipeline_logging.info(message)

            message = f"{self.operation}: Getting Validated Data"
            self.ccs_training_pipeline_logging.info(message)

            # GETTING THE DATA
            data_loader = CCSDataLoader(is_training=True)
            validated_data = data_loader.ccs_get_data()

            message = f"{self.operation}: Validated Data Obtained"
            self.ccs_training_pipeline_logging.info(message)

            # DATA PRE-PROCESSING

            message = f"{self.operation}: Data Preprocessing started"
            self.ccs_training_pipeline_logging.info(message)

            eda = CCSEda(is_training=True)
            feature_engineer = CCSFeatureEngineering(is_training=True)
            feature_selector = CCSFeatureSelection(is_training=True)
            file_operator = CCSFileOperations()

            temp_df = feature_selector.ccs_remove_columns(validated_data, 'id')
            message = f"{self.operation}: Removed the 'ID' column"
            self.ccs_training_pipeline_logging.info(message)

            features, label = eda.ccs_feature_label_split(temp_df, ['Concrete compressive strength(MPa, megapascals)'])
            message = f"{self.operation}: Separated the features and labels"
            self.ccs_training_pipeline_logging.info(message)

            is_null_present, columns_with_null = eda.ccs_features_with_missing_values(features)

            col_to_drop = []
            if is_null_present:
                features, dropped_features = feature_engineer.ccs_handling_missing_data_mcar(features,
                                                                                             columns_with_null)
                col_to_drop.extend(dropped_features)

            message = f"{self.operation}: Checked for null values and if any were present imputed them"
            self.ccs_training_pipeline_logging.info(message)

            cont_feat, discrete_feat = eda.ccs_continuous_discrete_variables(features, features.columns)
            message = f"{self.operation}: Obtained the Continuous and Discrete Features"
            self.ccs_training_pipeline_logging.info(message)

            features = eda.ccs_obtain_normal_features(features, cont_feat)
            message = f"{self.operation}: Converted all possible continuous columns to normal"
            self.ccs_training_pipeline_logging.info(message)

            # Feature Selection
            col_to_drop.extend(feature_selector.ee_features_with_zero_std(features))
            col_to_drop.extend(feature_selector.ccs_feature_not_important(features, label))
            col_to_drop.extend(feature_selector.ccs_col_with_high_correlation(features))

            col_to_drop = list(set(col_to_drop))
            col_to_drop_str = ",".join(col_to_drop)

            with open(os.path.join(self.rel_info_dir, "columns_to_drop.txt"), 'w') as f:
                f.write(col_to_drop_str)

            features = feature_selector.ccs_remove_columns(features, col_to_drop)
            message = f"{self.operation}: Dropped all the relevant columns after feature selection"
            self.ccs_training_pipeline_logging.info(message)

            features = feature_engineer.ccs_standard_scaling_features(features)

            message = f"{self.operation}: All the features have been scaled"
            self.ccs_training_pipeline_logging.info(message)

            message = f"{self.operation}: Data Preprocessing completed"
            self.ccs_training_pipeline_logging.info(message)

            # CLUSTERING

            message = f"{self.operation}: Data Clustering Started"
            self.ccs_training_pipeline_logging.info(message)

            cluster = CCSClusteringTrain()
            num_clusters = cluster.ccs_obtain_optimum_cluster(features)
            features = cluster.ccs_create_cluster(features, num_clusters)

            features['Concrete compressive strength(MPa, megapascals)'] = label

            list_of_cluster = features['cluster'].unique()

            message = f"{self.operation}: Data Clustering Completed"
            self.ccs_training_pipeline_logging.info(message)

            # Training of Each Cluster

            for i in list_of_cluster:
                message = f"{self.operation}: Start of Training for cluster {i}"
                self.ccs_training_pipeline_logging.info(message)

                cluster_data = features[features['cluster'] == i]
                cluster_feature = cluster_data.drop(columns=['Concrete compressive strength(MPa, megapascals)', 'cluster'])
                cluster_label = cluster_data['Concrete compressive strength(MPa, megapascals)']

                train_x, test_x, train_y, test_y = train_test_split(cluster_feature, cluster_label, random_state=42)
                model_finder = CCSModelFinderTrain()
                model_name, model = model_finder.ccs_best_model(train_x=train_x, train_y=train_y,
                                                                test_x=test_x, test_y=test_y)

                file_operator.ccs_save_model(model=model, model_dir=self.ml_model_dir,
                                             model_name=f"{model_name}_cluster_{i}.pickle")

                message = f"{self.operation}:Model for cluster {i} trained"
                self.ccs_training_pipeline_logging.info(message)

            message = f"{self.operation}: Successful End of Training "
            self.ccs_training_pipeline_logging.info(message)

            message = f"{self.operation}: Training Pipeline Successfully Completed"
            self.ccs_training_pipeline_logging.info(message)

        except Exception as e:
            message = f"{self.operation}: There was an ERROR in obtaining best model: {str(e)}"
            self.ccs_training_pipeline_logging.info(message)
            raise e
