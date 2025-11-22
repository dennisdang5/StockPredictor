import os

from google.cloud import bigquery

project_number = os.environ["multimodal-forecasting"]

client = bigquery.Client(project=project_number)