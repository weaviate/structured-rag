import weaviate
import weaviate.classes.config as wvcc
from weaviate.util import get_valid_uuid
from uuid import uuid4

from structured_rag.run_test.utils_and_metrics.helpers import load_json_from_file

import time
import requests

# import dataset to Weaviate
# load dataset
dataset_filepath = "../../../data/WikiQuestions.json"
dataset = load_json_from_file(dataset_filepath)

# connect to Weaviate and create collection
weaviate_client = weaviate.connect_to_local()

weaviate_client.collections.delete("StructuredRAG")

structured_rag_collection = weaviate_client.collections.create(
    name="StructuredRAG",
    properties=[
        wvcc.Property(name="title", data_type=wvcc.DataType.TEXT),
        wvcc.Property(name="context", data_type=wvcc.DataType.TEXT),
        wvcc.Property(name="question", data_type=wvcc.DataType.TEXT),
        wvcc.Property(name="answerable", data_type=wvcc.DataType.BOOL),
        wvcc.Property(name="answer", data_type=wvcc.DataType.TEXT),
        wvcc.Property(name="gfl_assessed_answerability", data_type=wvcc.DataType.BOOL),
    ]
)

# upload dataset to Weaviate

for row in dataset:
    id = get_valid_uuid(uuid4())
    structured_rag_collection.data.insert(
        properties=row,
        uuid=id,
    )

print(f"\033[92mDataset uploaded to Weaviate\033[0m")

# start timer
start_time = time.time()

# send to GFL

# poll GFL service until finished

# stop timer

# get results

# evaluate results