from dotenv import load_dotenv
from airflow.decorators import task, dag
from mlflow_provider.operators.registry import CreateRegisteredModelOperator
from sklearn.metrics import roc_auc_score, average_precision_score
from datetime import datetime
import logging
import pandas as pd
from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import StratifiedGroupKFold
from airflow.providers.google.cloud.operators.gcs import GCSCreateBucketOperator
from mlflow_provider.hooks.client import MLflowClientHook
import os
import mlflow
import sys
sys.path.append("/opt/airflow/")
from src.utils.metrics import early_enrichment, diverse_early_enrichment
from src.utils.model import AirCheckModel
from src.utils.logger import setup_logger
from src.utils.aircheck_io import read_aircheck_file

mlflow_enable_system_metrics = os.getenv('MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING', 'false').lower() == 'true'


os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"




load_dotenv(dotenv_path='/opt/airflow/.env')
MLFLOW_CONN_ID = os.getenv("MLFLOW_CONN_ID", "mlflow_default_new")
GCP_CONN_ID = os.getenv("GCP_CONN_ID", "google_cloud_default")
MAX_RESULTS_MLFLOW_LIST_EXPERIMENTS = int(
    os.getenv("MAX_RESULTS_MLFLOW_LIST_EXPERIMENTS", 100))
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "Del-ml")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "my_model")
ARTIFACT_BUCKET = os.getenv("ARTIFACT_BUCKET", "testmlflow")

DEFAULT_NUM_FOLDS = int(os.getenv("DEFAULT_NUM_FOLDS", 5))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 16))
INPUT_FILE = os.getenv("INPUT_FILE", './data/input_data/S202309_WDR12.tsv.gz')

COMPANY = os.getenv('COMPANY', 'hitgen')
# FPS = ["ECFP6", "ECFP4"]
FPS = os.getenv("FPS", ["ECFP6", "ECFP4"])
RUN_NAME = "SAMPLE_RUN"
n_splits = int(os.getenv("n_splits", 2))
INPUT_FILE_FORMAT = 'parquet'


@dag(
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    doc_md=__doc__
)
def create_new_del_model():

    logger = setup_logger("aircheck-cv", debug="Test")

    create_buckets_if_not_exists = GCSCreateBucketOperator(
        task_id="create_buckets_if_not_exists",
        gcp_conn_id=GCP_CONN_ID,
        bucket_name=ARTIFACT_BUCKET,
    )

    @task
    def create_experiment(experiment_name, artifact_bucket, **context):
       
        """Create a new MLFlow experiment with a specified name.
        Save artifacts to the specified S3 bucket."""

        ts = context["ts"]

        mlflow_hook = MLflowClientHook(mlflow_conn_id=MLFLOW_CONN_ID)
        new_experiment_information = mlflow_hook.run(
            endpoint="api/2.0/mlflow/experiments/create",
            request_params={
                "name": ts + "_" + experiment_name,
                "artifact_location": f"gs://{artifact_bucket}/",
            },
        ).json()

        return new_experiment_information["experiment_id"]

    @task
    def load_dataset(input_file: str):
        """Load dataset from the input file."""
        print("Columns---", FPS)
        print("connection id----", MLFLOW_CONN_ID)
        dataset = read_aircheck_file(
            input_file, company=COMPANY, fps=FPS, file_format=INPUT_FILE_FORMAT)[0]

        # Add debugging information
        logger.debug(f"Dataset loaded: {dataset}")

        if dataset is None:
            raise ValueError(f"Failed to load dataset from {input_file}")

        logger.debug(f"Loaded {len(dataset.y)} datapoints from {input_file}")
      

        # Check dataset attributes before calling generate_groups
        if not hasattr(dataset, 'data') or not hasattr(dataset, 'y'):
            raise ValueError(
                "Dataset does not have the required attributes: data, y")

        dataset.generate_groups()

        # Add debugging information after calling generate_groups
        logger.debug(f"Dataset after generate_groups: {dataset}")

        if not hasattr(dataset, 'data') or not hasattr(dataset, 'y') or not hasattr(dataset, 'groups'):
            raise ValueError(
                "Dataset does not have the required attributes: data, y, groups")

        return {
            'data': dataset.data.tolist(),
            'y': dataset.y.tolist(),
            'groups': dataset.groups.tolist()}

    @task
    def lgbm_classifier_model(experiment_id: str, dataset: dict, **kwargs):
        """Train and validate model using a lgbm.

        Returns accuracy score via XCom to GCS bucket.
        """

        # mlflow.set_tracking_uri('http://34.139.49.139:5000/')
        os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
        mlflow.set_tracking_uri('http://34.139.49.139:5000/')
        mlflow.sklearn.autolog()
        mlflow.lightgbm.autolog()

        X = pd.DataFrame(dataset['data'])
        y = pd.Series(dataset['y'])
        groups = pd.Series(dataset['groups'])
        groups = groups.to_numpy()

        experiment_id = kwargs['ti'].xcom_pull(
            task_ids='create_experiment', key='return_value')
        overall_metrics = {}

        with mlflow.start_run(experiment_id=experiment_id, run_name=f'lgbm_{kwargs["run_id"]}'):

            logging.info('Performing lgbm')
            mlflow.set_tag("model_type", "lgbm")

            s = StratifiedGroupKFold(n_splits=int(
                n_splits), shuffle=True, random_state=42)
            for fold, (train_idx, test_idx) in enumerate(s.split(X=X, y=y, groups=groups)):

                with mlflow.start_run(run_name=f"fold_{fold}", nested=True) as run2:
                    logger.info(f"starting cv fold {fold}")

                    model = AirCheckModel(base_model=LGBMClassifier(
                        random_state=42, n_jobs=NUM_WORKERS), fp="ECFP4")
                    logger.info(f"using lgbm model")

                    train_X, train_y = X.iloc[train_idx], y.iloc[train_idx]
                    test_X, test_y = X.iloc[test_idx], y.iloc[test_idx]

                    model.fit(train_X, train_y)

                    y_pred = model.predict_proba(test_X)
                    test_y = test_y.to_numpy()

                    metrics = {
                        "roc_auc": roc_auc_score(test_y, y_pred[:, 1]),
                        "average_precision": average_precision_score(test_y, y_pred[:, 1]),
                        "early_enrichment": early_enrichment(test_y, y_pred[:, 1], _top_n=200),
                        "diverse_early_enrichment": diverse_early_enrichment(test_y, y_pred[:, 1], groups[test_idx]),
                    }

                    mlflow.log_metrics(metrics)

                    for key, val in metrics.items():
                        overall_metrics.setdefault(key, []).append(val)

            cv_df = pd.DataFrame(overall_metrics)
            cv_mean = {"cv_avg_" + key: val for key,
                       val in cv_df.mean().to_dict().items()}
            cv_std = {"cv_std_" + key: val for key,
                      val in cv_df.std().to_dict().items()}
            mlflow.log_metrics(cv_mean)
            mlflow.log_metrics(cv_std)
            logging.info('Training model with best parameters')

    create_registered_model = CreateRegisteredModelOperator(
        task_id="create_registered_model",
        name="{{ ts }}" + "_" + REGISTERED_MODEL_NAME,
        tags=[
            {"key": "model_type", "value": "regression"},
            {"key": "data", "value": "dna_encoded"},
        ],
        mlflow_conn_id=MLFLOW_CONN_ID,
    )

    dataset = load_dataset(INPUT_FILE)
    experiment_created = create_experiment(
        experiment_name=EXPERIMENT_NAME, artifact_bucket=ARTIFACT_BUCKET
    )
    (
        create_buckets_if_not_exists
        >> experiment_created
        >> lgbm_classifier_model(experiment_id=experiment_created, dataset=dataset)
        >> create_registered_model
    )


create_new_del_model()
