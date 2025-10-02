# Movie Recommendation with SAR

This directory contains a Python script (`movie_recommendation.py`) that demonstrates how to build a simple movie recommender system using the SAR (Simple Algorithm for Recommendations) algorithm.

## What it does

The script performs the following steps:
1.  **Loads Data**: It downloads and loads the MovieLens 100k dataset.
2.  **Splits Data**: It splits the data into training and testing sets.
3.  **Trains Model**: It trains a SAR model on the training data. The SAR model is a fast and scalable algorithm based on item co-occurrence.
4.  **Generates Recommendations**: It predicts the top 10 movie recommendations for users in the test set.
5.  **Evaluates Model**: It evaluates the quality of the recommendations using standard ranking metrics like Precision@k, Recall@k, NDCG@k, and MAP.

## How to Run

1.  **Install Dependencies**:
    First, install the required Python packages from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Script**:
    Execute the main script from the terminal.
    ```bash
    python movie_recommendation.py
    ```

The script will print model training and prediction times, a sample of the recommendations, and the final evaluation metrics to the console.