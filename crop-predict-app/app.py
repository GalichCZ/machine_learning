from fastapi import FastAPI
from typing import List
import kagglehub
import pandas as pd
import numpy as np
import json
import os
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(title="Crop Recommendation API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Define input model
class PredictionInput(BaseModel):
    features: List[float]


# Check if weights file exists and load, or train model
def initialize_model():
    global weights, scaler, le, encoder

    # Download dataset
    path = kagglehub.dataset_download("atharvaingle/crop-recommendation-dataset")
    print("Path to dataset files:", path)

    # Load the dataset
    df = pd.read_csv(f"{path}/Crop_recommendation.csv")
    print("Dataset loaded. First few rows:")
    print(df.head())

    # Preprocess data
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])  # Converts crop names to integers

    # Standardize input features
    scaler = StandardScaler()
    x = scaler.fit_transform(df.drop(columns=['label']))

    # One-hot encode labels for multi-class classification
    encoder = OneHotEncoder(sparse_output=False)
    y_one_hot = encoder.fit_transform(df[['label']])  # Convert labels to one-hot

    # Convert to float
    x = x.astype(float)
    y_one_hot = y_one_hot.astype(float)

    print(f"Data preprocessed. Features shape: {x.shape}, Labels shape: {y_one_hot.shape}")

    # Check if weights file exists
    if os.path.exists("weights.json"):
        print("Loading existing weights from weights.json")
        with open("weights.json", "r") as f:
            weights_dict = json.load(f)
            weights = np.array(weights_dict["weights"])
    else:
        print("No existing weights found. Training model...")
        num_inputs = 7
        num_classes = 22
        weights = np.random.randn(num_inputs, num_classes) * 0.01  # Small random weights
        train_model(weights, x, y_one_hot)

    return weights, scaler, le


def outer_product(vec_a, vec_b):
    assert len(vec_a) > 0 and len(vec_b) > 0, "Input vectors must not be empty"

    out = np.zeros((len(vec_a), len(vec_b)))

    for i in range(len(vec_a)):  # Iterate over vec_a (rows)
        for j in range(len(vec_b)):  # Iterate over vec_b (columns)
            out[i][j] = vec_a[i] * vec_b[j]  # Multiply element-wise

    return out


def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Stabilize softmax
    return exp_x / np.sum(exp_x)  # Convert scores into probabilities


def neural_network(input, weights):
    return softmax(np.dot(input, weights))  # Matrix multiplication + softmax


def train_model(weights, x, y_one_hot):
    alpha = 0.01  # Learning rate
    epochs = 100

    for epoch in range(epochs):
        total_loss = 0

        for i in range(len(x)):  # Loop over dataset
            input = x[i]  # Features
            true = y_one_hot[i]  # One-hot encoded label

            # Forward pass
            pred = neural_network(input, weights)

            # Compute loss (cross-entropy)
            loss = (pred - true) ** 2  # Simple squared error
            total_loss += np.sum(loss)

            # Compute gradient (backpropagation)
            delta = pred - true  # Gradient of softmax + cross-entropy
            weight_deltas = outer_product(input, delta)  # Outer product

            # Update weights
            weights -= alpha * weight_deltas

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {total_loss}")

    # Save weights
    weights_dict = {"weights": weights.tolist()}  # Convert numpy array to list for JSON
    with open("weights.json", "w") as f:
        json.dump(weights_dict, f)

    print("Final weights saved to weights.json")
    return weights


def predict_crop(input_features):
    input_features = scaler.transform([input_features])  # Normalize input
    probs = neural_network(input_features[0], weights)  # Get probability distribution

    crop_names = le.inverse_transform(np.arange(len(probs)))  # Get all crop names

    # Calculate percentages and format them
    predictions = [
        {
            "crop": str(crop),
            "probability": float(prob),
            "percentage": f"{float(prob) * 100:.2f}%"  # Format as percentage with 2 decimal places
        }
        for crop, prob in zip(crop_names, probs)
    ]

    # Sort by highest probability
    sorted_predictions = sorted(predictions, key=lambda x: x["probability"], reverse=True)

    # Optionally filter out very small percentages (e.g., less than 0.01%)
    filtered_predictions = [pred for pred in sorted_predictions if pred["probability"] >= 0.0001]

    return filtered_predictions

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    global weights, scaler, le
    weights, scaler, le = initialize_model()
    print("Model initialization complete")


@app.post("/predict")
async def predict(input_data: PredictionInput):
    if len(input_data.features) != 7:
        return {"error": "Input must contain exactly 7 features"}

    predictions = predict_crop(input_data.features)
    return {"predictions": predictions}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# If you want to manually specify weights from a JSON string
@app.post("/set-weights")
async def set_weights(weights_json: str):
    global weights
    weights_dict = json.loads(weights_json)
    weights = np.array(weights_dict["weights"])
    return {"message": "Weights updated successfully"}


# For running directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)