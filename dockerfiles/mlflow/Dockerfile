FROM python:3.11-slim-buster
RUN apt-get update
RUN apt-get install -y postgresql postgresql-contrib postgresql-server-dev-all python-psycopg2
COPY dockerfiles/mlflow/requirements.txt requirements.txt
RUN pip install -r requirements.txt