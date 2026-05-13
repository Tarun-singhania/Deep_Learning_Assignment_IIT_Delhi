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

    os.makedirs(args.predictions_path, exist_ok=True)

    # Load test data
    df_test = pd.read_csv(args.test_data)
    df_test = preprocess_data(df_test)

    X_test = df_test.to_numpy(dtype=np.float32)

    # Load preprocessing params
    with open(args.pre_processing_params_path, "r") as f:
        params = json.load(f)

    mean = np.array(list(params["feature_mean"].values()), dtype=np.float32)
    std  = np.array(list(params["feature_std"].values()), dtype=np.float32)

    X_test = (X_test - mean) / std

    # Load trained model
    model = MyRegressor(
    n_features=X_test.shape[1],
    hidden_layers=[],
    activation_function="relu"
)

    model.load_model(args.model_path)

    # Inference
    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log).reshape(-1)

    # Save predictions
    output_path = os.path.join(args.predictions_path, "predictions.csv")

    pd.DataFrame(preds).to_csv(
        output_path,
        index=False,
        header=False
    )

    print(f"predictions.csv saved at {output_path}")


if __name__ == "__main__":
    main()
