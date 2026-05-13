import argparse
import os
import json
import numpy as np
import pandas as pd

from nn_from_scratch import MyRegressor
from train_model import preprocess_data


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--pre_processing_params_path", required=True)
    parser.add_argument("--predictions_path", required=True)

    args = parser.parse_args()

    # Create predictions directory if it does not exist
    if not os.path.exists(args.predictions_path):
        os.makedirs(args.predictions_path)

    # Load test data
    df_test = pd.read_csv(args.test_data)

    df_test = preprocess_data(df_test)

    X_test = df_test.to_numpy(dtype=np.float32)

    # Load preprocessing parameters
    with open(args.pre_processing_params_path, "r") as f:
        params = json.load(f)

    mean = np.array(list(params["feature_mean"].values()), dtype=np.float32)
    std = np.array(list(params["feature_std"].values()), dtype=np.float32)

    X_test = (X_test - mean) / std

    # Get architecture from model filename
    filename = os.path.basename(args.model_path)

    try:
        parts = filename.replace(".npz", "").split("_")
        hidden_layers = [int(x) for x in parts[2:-1]]
        activation = parts[-1]
    except:
        raise ValueError("Model filename format incorrect. Expected model_hidden_<sizes>_<activation>.npz")

    # Load trained model
    model = MyRegressor(
        n_features=X_test.shape[1],
        hidden_layers=hidden_layers,
        activation_function=activation
    )

    model.load_model(args.model_path)

    # Run inference
    preds_log = model.predict(X_test)

    preds = np.expm1(preds_log).reshape(-1)

    # Save predictions
    output_path = os.path.join(args.predictions_path, "predictions.csv")

    pd.DataFrame(preds).to_csv(
        output_path,
        index=False,
        header=False
    )


if __name__ == "__main__":
    main()