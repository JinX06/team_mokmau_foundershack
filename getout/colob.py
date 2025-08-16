from google.cloud import aiplatform

PROJECT_ID = "gen-lang-client-0182072294"
REGION = "us-central1"
MODEL_ID = "505997696619"  # Replace with your model ID

# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=REGION)

# Get the model
model = aiplatform.Model(model_name=MODEL_ID)

# Sample input for prediction
instances = [
    {"input_text": "Hello, world!"}
]

# Make prediction
prediction = model.predict(instances)
print(prediction)
