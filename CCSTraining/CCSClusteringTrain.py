import os
import logging
from kneed import KneeLocator
from sklearn.cluster import KMeans
from CCSCommonTasks.CCSFileOperations import CCSFileOperations


class CCSClusteringTrain:
    """
    :Class Name: CCSClusteringTrain
    :Description: This class is used to cluster the data so that models will be fine
                  tuned for each cluster and higher accuracy is obtained.

    Written By: Jobin Mathew
    Interning at iNeuron Intelligence
    Version: 1.0
    """

    def __init__(self):
        """
        :Method Name: __init__
        :Description: This method is Constructor for class EEFeatureSelectionTrain.
                      Initializes variables for logging
        """

        self.operation = 'TRAINING'

        self.file_operator = CCSFileOperations()

        if not os.path.isdir("CCSLogFiles/training/"):
            os.mkdir("CCSLogFiles/training/")
        self.log_path = "CCSLogFiles/training/CCSClusteringTrain.txt"

        self.ccs_clustering_logging = logging.getLogger("ccs_clustering_log")
        self.ccs_clustering_logging.setLevel(logging.INFO)
        ccs_clustering_handler = logging.FileHandler(self.log_path)
        formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s',
                                      datefmt='%m/%d/%Y %I:%M:%S %p')
        ccs_clustering_handler.setFormatter(formatter)
        self.ccs_clustering_logging.addHandler(ccs_clustering_handler)

        self.cluster_model_path = "CCSModels/"

    def ccs_obtain_optimum_cluster(self, dataframe):
        """
        :Method Name: ccs_obtain_optimum_cluster
        :Description: This method calculates the optimum no. of cluster

        :param dataframe: The dataframe representing the data from the client after
                          all the preprocessing has been done
        :return: The optimum cluster value
        :On Failure: Exception
        """
        try:
            # within cluster sum of squares: For evaluating the knee point so that no. of clusters can be determined
            wcss = []

            for i in range(1, 11):
                # initializer is k-means++ so that there is some minimum distance between randomly initialized centroid.
                kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
                kmeans.fit(dataframe)
                wcss.append(kmeans.inertia_)

            # KneeLocator mathematically determines the knee point so that the task of selecting the optimum no of
            # cluster can automated.
            opt_cluster_val = KneeLocator(range(1, 11), wcss, curve="convex", direction='decreasing').knee

            message = f"{self.operation} The optimum cluster value obtained is {opt_cluster_val} with wcss={wcss[opt_cluster_val-1]}"
            self.ccs_clustering_logging.info(message)

            return opt_cluster_val

        except Exception as e:
            message = f"{self.operation}: There was an ERROR while detecting optimum no. of cluster: {str(e)}"
            self.ccs_clustering_logging.error(message)
            raise e

    def ccs_create_cluster(self, dataframe, number_of_clusters):
        """
        :Method Name: ccs_create_cluster
        :Description: This method performs the clustering in the dataset after preprocessing.

        :param dataframe: The pandas dataframe which has to be clustered.
        :param number_of_clusters: The number of clusters the data has to be clustered into.
        :return: Dataframe with clusters number added as a new column.
        :return: The sklearn Model used for clustering.
        :On Failure: Exception
        """
        try:

            k_mean_model = KMeans(n_clusters=number_of_clusters, init='k-means++', random_state=42)

            # Adds a new column to the dataframe which identifies the cluster to which that data point belongs to.
            dataframe['cluster'] = k_mean_model.fit_predict(dataframe)

            self.file_operator.ccs_save_model(k_mean_model, self.cluster_model_path, "cluster.pickle")

            message = f"{self.operation}: EEClustering has been done, with cluster column added to dataset"
            self.ccs_clustering_logging.info(message)

            return dataframe

        except Exception as e:
            message = f"There was an ERROR while creating cluster: {str(e)}"
            self.ccs_clustering_logging.error(message)
            raise e
