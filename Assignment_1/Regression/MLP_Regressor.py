import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from train_model import preprocess_data
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

PLOT_DIR = "Images_Plot_MLP_Regressor"

# load and split the train and val dataset respectively
df_train = pd.read_csv("train.csv") 
df_val=pd.read_csv("val.csv")

# Preprocess data
df_train = preprocess_data(df_train)
df_val = preprocess_data(df_val)

# Split inputs & outputs
df_train_inputs = df_train.drop(columns=['trip_duration'])
df_train_output = np.log1p(df_train['trip_duration'])
df_val_inputs = df_val.drop(columns=['trip_duration'])
df_val_output = np.log1p(df_val['trip_duration'])


# Scaling 
scaler = StandardScaler()

df_train_inputs = scaler.fit_transform(df_train_inputs)
df_val_inputs = scaler.transform(df_val_inputs)

y_mean, y_std = df_train_output.mean(), df_train_output.std()
df_train_output = (df_train_output - y_mean) / y_std
df_val_output = (df_val_output - y_mean) / y_std


# Experimental Architectures
architectures = {
    "1_hidden_layer": (256,),
    "2_hidden_layers": (256, 128),
    "3_hidden_layers": (256, 128, 64),
    "4_hidden_layers":(256,128,64,32)
}

def run_experiments(epochs=30):
    results = {}
    for name, hidden_layers in architectures.items():
        print(f"\nTraining model: {name} with activation relu")

        model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="sgd",
            learning_rate_init=0.00001,
            batch_size=32,
            max_iter=1,          # train one epoch at a time
            warm_start=True,     # continue training across epochs
            random_state=42
        )

        train_losses, val_losses = [], []
        start_time = time.time()

        for epoch in range(epochs):
            # Train for one epoch
            model.fit(df_train_inputs, df_train_output)

            # Record training loss (last entry of loss_curve_)
            train_losses.append(model.loss_curve_[-1])

            # Compute validation loss
            val_pred = model.predict(df_val_inputs)
            val_loss = mean_squared_error(df_val_output, val_pred)
            val_losses.append(val_loss)

        training_time = time.time() - start_time

        results[name] = {
            "training_loss": train_losses,
            "validation_loss": val_losses,
            "total_time": training_time
        }

        print(f"Training time: {training_time:.2f} seconds")

    return results


# Run experiments for ReLU
results_relu = run_experiments()

# Plot Training Loss for all given architectures
plt.figure(figsize=(12, 6))
for name, res in results_relu.items():
    plt.plot(res["training_loss"], label=f"{name} - Train")
plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.title("Training Loss vs Epochs (ReLU)")
plt.legend()
plt.savefig(os.path.join(PLOT_DIR, "train_loss_relu.png"),dpi=300, bbox_inches="tight")
plt.close()

# Plot Validation Loss for all given architectures
plt.figure(figsize=(12, 6))
for name, res in results_relu.items():
    plt.plot(res["validation_loss"], label=f"{name} - Validation")
plt.xlabel("Epochs")
plt.ylabel("Validation Loss (MSE)")
plt.title("Validation Loss vs Epochs (ReLU)")
plt.legend()
plt.savefig(os.path.join(PLOT_DIR, "val_loss_relu.png"),dpi=300, bbox_inches="tight")
plt.close()

# Print Training Times
print("\nTraining Times Comparison:")
for name, res in results_relu.items():
    print(f"{name}: ReLU={res['total_time']:.2f}s")