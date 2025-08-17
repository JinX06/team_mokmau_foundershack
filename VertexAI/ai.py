from google.cloud import aiplatform

import os
import uuid
import json
import numpy as np
import pandas as pd
from typing import Dict, Callable, List, Union

from dotenv import load_dotenv
from google.cloud import storage
from google.cloud import aiplatform

# ========= Config / setup =========
load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION", "us-central1")
BUCKET = os.getenv("BUCKET")                  # no "gs://" prefix; just the bucket name
GCS_PREFIX = os.getenv("GCS_PREFIX", "gotcha/datasets")
project = PROJECT_ID
location = REGION
endpoint_id = os.getenv("ENDPOINT_ID")  # replace with your endpoint ID

instance = {
    "student_id": "stu_999",
    "submission_id": "sub_999",
    "text_excerpt": "In this experiment, we analyzed three variables and observed significant improvements.",
    "notes": "Formal, structured, technical vocabulary",
    "avg_sentence_length": "22.5",
    "type_token_ratio": 0.52,
    "formality_score": 0.80,
    "transitional_phrase_rate": 7.1,
    "repetitiveness_score": 0.35,
    "personalization_score": 0.10,
    "gltr_like_uniformity": 0.70,
    "timestamp": "2025-08-17T10:00:00Z",
    "phase": "TEST",
    "topic": "SCIENCE",   # <-- ADD THIS
}


# NOTE: AutoML inferred several numeric-looking columns as STRING during training.
# Prediction payloads must match EXACT types. Cast the following to string:
STRING_COLS = [
    "avg_sentence_length",
    "type_token_ratio",
    "formality_score",
    "transitional_phrase_rate",
    "repetitiveness_score",
    "personalization_score",
    "gltr_like_uniformity",
    "embedding_similarity_to_baseline",
    "time_to_submit_hours",
]
# Provide safe defaults for missing numeric-looking features inferred as STRING
if instance.get("embedding_similarity_to_baseline") is None:
    instance["embedding_similarity_to_baseline"] = "0.0"
if instance.get("time_to_submit_hours") is None:
    instance["time_to_submit_hours"] = "0.0"
for col in STRING_COLS:
    if col in instance and instance[col] is not None:
        instance[col] = str(instance[col])

# Ensure timestamp is a string (AutoML datetime feature expects RFC3339 string)
if instance.get("timestamp") is not None:
    instance["timestamp"] = str(instance["timestamp"])

# Ensure phase is a string (AutoML categorical feature)
if instance.get("phase") is not None:
    instance["phase"] = str(instance["phase"])

from google.cloud import aiplatform_v1

# Ensure required categorical fields exist (adjust to your training schema)
if instance.get("difficulty") is None:
    instance["difficulty"] = "MEDIUM"
else:
    instance["difficulty"] = str(instance["difficulty"])  # cast to string if present

# Build full resource name for the endpoint and call the PredictionService API directly
endpoint_name = f"projects/{project}/locations/{location}/endpoints/{endpoint_id}"
client = aiplatform_v1.PredictionServiceClient()
try:
    prediction = client.predict(endpoint=endpoint_name, instances=[instance])
    print("Prediction:", prediction)
except Exception as e:
    msg = str(e)
    print("Prediction failed:", msg)
    if "Missing struct property" in msg:
        import re
        m = re.search(r"Missing struct property: (\w+)", msg)
        if m:
            print("Action: add required feature to instance:", m.group(1))
    raise