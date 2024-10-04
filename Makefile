# https://medium.com/@rodelvanrooijen/airflow-on-gke-using-helm-15ca05c11364
# https://towardsdatascience.com/deploying-airflow-on-google-kubernetes-engine-with-helm-part-two-f833b0a3b0b1



gcloud iam service-accounts create airflow-kubernetes \
  --description="User-managed service account for the Airflow deployment" \
  --display-name="Airflow Kubernetes"


gcloud projects add-iam-policy-binding aircheck-398319 \
  --member=serviceAccount:airflow-kubernetes@aircheck-398319.iam.gserviceaccount.com \
  --role=roles/container.admin \
  --role=roles/iam.serviceAccountUser \
  --role=roles/iam.workloadIdentityUser \
  --role=roles/storage.admin

gcloud container clusters create airflow-cluster \
  --zone "us-east1-b" \
  --project aircheck-398319 \
  --machine-type n2-standard-4 \
  --num-nodes 1 \
  --scopes "cloud-platform" \
  --autoscaling-profile "optimize-utilization" \
  --enable-autoscaling --min-nodes=1 --max-nodes=3 \
  --workload-pool aircheck-398319.svc.id.goog \
  --no-enable-ip-alias


gcloud container clusters get-credentials airflow-cluster \
      --zone "us-east1-b" \
      --project aircheck-398319

kubectl create ns "airflow"


helm create mlflow




gcloud iam service-accounts add-iam-policy-binding airflow-kubernetes@aircheck-398319.iam.gserviceaccount.com \
    --role roles/iam.workloadIdentityUser \
    --member "serviceAccount:aircheck-398319.svc.id.goog[airflow/airflow]"

gcloud compute addresses create airflow-api-ip \
    --region us-east1 \
    --subnet default

gcloud compute addresses describe airflow-api-ip


gcloud artifacts repositories create airflow-custom-images \
    --repository-format=docker \
    --location=us-east1 \
    --description="Airflow docker image repository" \
    --project=aircheck-398319


docker build . \
  -f dockerfiles/airflow/Dockerfile \
  -t us-east1-docker.pkg.dev/aircheck-398319/airflow-custom-images/airflow-custom:0.0.1 \
  --platform=linux/amd64
docker push us-east1-docker.pkg.dev/aircheck-398319/airflow-custom-images/airflow-custom:0.0.1


docker build . \
  -f dockerfiles/mlflow/Dockerfile \
  -t us-east1-docker.pkg.dev/aircheck-398319/airflow-custom-images/mlflow-custom:0.0.1 \
  --platform=linux/amd64

docker push us-east1-docker.pkg.dev/aircheck-398319/airflow-custom-images/mlflow-custom:0.0.1

gcloud auth configure-docker us-east1-docker.pkg.dev



gcloud sql instances create postgres-airflow_test \
      --database-version=POSTGRES_16 \ 
      --cpu=2 \
      --memory=8GB \
      --region=us-east1 \
      --no-assign-ip \
      --enable-google-private-path \
      --root-password=test@aircheck


gcloud sql instances create airflow-test \
      --database-version=POSTGRES_15 \
      --cpu=2 \
      --memory=8GB \
      --region=us-east1 \
      --network=default \
      --no-assign-ip \
      --root-password=test@aircheck




# helm repo add airflow-stable https://airflow-helm.github.io/charts

helm repo add apache-airflow https://airflow.apache.org
helm upgrade --install airflow apache-airflow/airflow --namespace airflow --create-namespace
helm repo update


gcloud compute disks create --size=10GB --zone=us-east1-b airflow-dags-disk

kubectl apply -f ./postgres-secrets.yaml -n "airflow"
kubectl apply -f secrets/secret-redis-password.yaml -n "airflow"

kubectl apply -f secrets/postgres-secrets.yaml -n "airflow"
kubectl apply -f secrets/project_variables.yaml -n "airflow"
kubectl apply -f secrets/secret-fernet-key.yaml -n "airflow"
kubectl apply -f secrets/secret-webserver-key.yaml -n "airflow"

kubectl apply -f persistent-volumes.yaml -n "airflow"


helm upgrade --install airflow apache-airflow/airflow -n airflow  \
  -f values.yaml \
  --debug



helm install \
  deploy-airflow \
  airflow-stable/airflow \
  --namespace airflow \
  --version "8.9.0" \
  --values custom-values.yaml \
  --debug


helm upgrade \
  deploy-airflow \
  airflow-stable/airflow \
  --namespace airflow \
  --version "8.9.0" \
  --values values.yaml \
  --debug


  helm install \
  deploy-airflow \
  airflow-stable/airflow \
  --namespace airflow \
  --version "8.9.0" \
  --values values.yaml \
  --debug


  # get namespace:
  kubectl get pods --namespace <name>
   helm list --all-namespaces 

   helm uninstall deploy-airflow --namespace airflow


  #  log
  kubectl get pods --namespace <name>
  kubectl logs deploy-airflow-webserver-xxxx --namespace airflow
  kubectl get events -n airflow --sort-by=.metadata.creationTimestamp
  kubectl get secrets -n airflow
kubectl describe secret airflow-cluster-postgres-credentials -n airflow
kubectl describe pod deploy-airflow-db-migrations-df8cc4989-rqw4x -n airflow


kubectl describe pod <pod-name> -n <namespace>

kubectl port-forward svc/airflow-web 8080:443 --namespace airflow 2>&1 >/dev/null &

kubectl logs deploy-airflow-worker-0 -n airflow -c check-db

kubectl run -it --rm --image=postgres:13 --restart=Never postgres-test -- psql -h 10.70.241.3 -U postgres -d airflow



gcloud container clusters describe airflow-cluster \
    --location=us-east1-b \
    --project=aircheck-398319 \
    --format "flattened(masterAuthorizedNetworksConfig.cidrBlocks[])"



    helm list -n airflow
    helm uninstall <release-name> -n <namespace>


    helm delete airflow --namespace airflow
    helm search repo


  * Airflow Webserver:  kubectl port-forward svc/deploy-airflow-web 8080:8080 --namespace airflow
  * Flower Dashboard:   kubectl port-forward svc/deploy-airflow-flower 5555:5555 --namespace airflow


  helm repo remove stable



  helm upgrade --install airflow apache-airflow/airflow -n airflow --debug


  helm show values apache-airflow/airflow > test_val.yaml



helm upgrade deploy-airflow  airflow-stable/airflow --namespace airflow \
  --values custom-values.yaml \
  --debug


kubectl port-forward svc/deploy-airflow-web 8080:443 --namespace airflow 