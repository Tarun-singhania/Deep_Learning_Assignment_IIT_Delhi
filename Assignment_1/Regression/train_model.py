import numpy as np
import pandas as pd
import os
import math
import argparse

import matplotlib.pyplot as plt

import json
from nn_from_scratch import MyRegressor   # import your class

def preprocess_data(Dataset):

  df = Dataset.copy()

    # For underStanding purpose:
#   # Shape of train dataset:
#   df.shape

#   # How dataset looks like:
#   df.head()

#   # datatypes of cols:
#   df.info()

#   # find missing value:
#   df.isnull().sum()

#   # relation between cols:
#   df.corr(numeric_only=True)

#   # Check duplicated values or not:
#   df.duplicated().sum()

  def check_weekend(day):
    return int((day==6) or (day==7))

  def rush_hour(hour):
    return int((hour>=7 and hour<=10) or (hour>=17 and hour<=21))

  # Convert into date_time object because we want to extract something from datetime col
  df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

  # Find day of week
  df["day_name"] = df['pickup_datetime'].dt.day_name()
  df['day_of_week'] = df['pickup_datetime'].dt.day_of_week +1   # 1 - Monday and 7 - Sunday

  # Find Weekend or not
  df['Weekend'] = df['day_of_week'].map(check_weekend).astype(np.int8)

  # Find day of hours
  df['day_of_hours'] = df['pickup_datetime'].dt.hour
  df['rush_hour'] = df['day_of_hours'].map(rush_hour).astype(np.int8)

  # Find date
  df['pickup_date'] = df['pickup_datetime'].dt.date

  # Convert Co-ordinates(degree) to radians
  def convert_into_radian(degree):
    return ((degree*(math.pi))/(180))

  # Apply Conversion in radians method on Co-ordinates cols
  df['pickup_longitude_rad'] = df['pickup_longitude'].map(convert_into_radian)
  df['pickup_latitude_rad'] = df['pickup_latitude'].map(convert_into_radian)
  df['dropoff_longitude_rad'] = df['dropoff_longitude'].map(convert_into_radian)
  df['dropoff_latitude_rad'] = df['dropoff_latitude'].map(convert_into_radian)

  # Create a method to calculate trip distance
  def distance(pick_lon,pick_lat,drop_lon,drop_lat):
    a = ((math.sin((drop_lat-pick_lat)/2))**2) + math.cos(pick_lat)*math.cos(drop_lat)*((math.sin((drop_lon-pick_lon)/2))**2)
    c = 2 * math.atan2(math.sqrt(a),math.sqrt(1-a))
    R = 6371     # Radius of Earth in KM
    distance = R*c
    return distance

  # Calculate trip distance
  df['trip_distance(KM)'] = list(
      map(distance,
          df['pickup_longitude_rad'],
          df['pickup_latitude_rad'],
          df['dropoff_longitude_rad'],
          df['dropoff_latitude_rad'])
  )

  df['trip_distance(KM)'] = df['trip_distance(KM)'].astype(np.float32)

  df['store_and_fwd_flag'] = df['store_and_fwd_flag'].map({'Y':1,'N':0}).astype(np.int8)

  # Calculate daily trip counts and It contains in Each row as total trips on that day
  daily_trips = df.groupby('pickup_date').size().reset_index(name='trip_count')

  # Calculate mean and standard deviation
  mean_trips = daily_trips['trip_count'].mean()
  std_trips = daily_trips['trip_count'].std()

  # this is my threshold I checked 1,2,3 and choose 1.5
  threshold = 1.5

  # Below lower_bound -> unusually low traffic
  # Above upper_bound -> unusually high traffic
  lower_bound = mean_trips - threshold * std_trips
  upper_bound = mean_trips + threshold * std_trips

  # High anomaly -> festivals, strikes ending, storms, events
  # Low anomaly -> lockdowns, extreme weather, holidays
  high_anomaly = daily_trips[daily_trips['trip_count'] > upper_bound]
  low_anomaly = daily_trips[daily_trips['trip_count'] < lower_bound]

  # Combine high and low anomalies
  anomalous_days = pd.concat([high_anomaly, low_anomaly])

  # Normal day as '0'
  df['is_anomaly'] = 0

  # anomalous day as '1'
  df.loc[df['pickup_date'].isin(anomalous_days['pickup_date']), 'is_anomaly'] = 1

  df["is_anomaly"] = df["is_anomaly"].astype(np.int8)

  # Calculate speed (distance in km / duration in hours)
  # Calculate speed (ONLY if trip_duration exists)
  if 'trip_duration' in df.columns:
    df = df[df['trip_duration'] > 0].copy()    # protect against inf
    df['speed'] = df['trip_distance(KM)'] / (df['trip_duration'] / 3600)
  else:
    # Dummy speed for test data (not used further)
    df['speed'] = 0.0


  # Calculate average speed for each day of week for each hour
  df_avg = df.groupby(['day_of_week', 'day_of_hours'])['speed'].mean().reset_index()

  # # Visualize to make decision for traffic
  #fig, ax = plt.subplots(figsize=(10, 10))
  #df_temp = df_avg.pivot(index='day_of_hours', columns='day_of_week', values='speed')
  #sns.heatmap(df_temp, annot=True, ax=ax)
  #plt.show()

  # We create a column for traffic is more or less
  # Initially,we let there is no traffic,it means all value of col contain 0
  df['is_traffic'] = 0

  # On the basic of avg speed of every day for each hour(By visulization with heatmap),we decide this
  df.loc[
      (((df['day_of_hours']>=8) & (df['day_of_hours'] <= 18)) &
       ((df['day_of_week']>=1) & (df['day_of_week'] <= 4))),
      'is_traffic'
  ] = 1
  df['is_traffic'] = df['is_traffic'].astype(np.int8)
  # Check unique values and their counts
  passenger_counts = df['passenger_count'].value_counts().sort_index()

  # I decided to remove 7, 8, 9 based on the counts very small, and I keep 0 because it could be realistic scenario
  # Filter for common values and decide outliers
  outliers = [7, 8, 9]
  df = df[~df['passenger_count'].isin(outliers)].copy()

  # Trips starting/ending inside NYC , Detect outliers / bad GPS points , Improve fare prediction , trip duration models &Filter unrealistic trips

  # Any GPS point inside this box is considered inside boundary.
  manhattan_bounds = {
      'lat_min': 40.70,
      'lat_max': 40.88,
      'lon_min': -74.02,
      'lon_max': -73.90
  }

  df['pickup_in_range'] = (
      (df['pickup_latitude'].between(40.70, 40.88)) &
      (df['pickup_longitude'].between(-74.02, -73.90))
  ).astype(np.int8)

  df['dropoff_in_range'] = (
      (df['dropoff_latitude'].between(40.70, 40.88)) &
      (df['dropoff_longitude'].between(-74.02, -73.90))
  ).astype(np.int8)

  df = df.drop(
    columns=[
        'id', 'day_name', 'pickup_datetime', 'dropoff_datetime',
        'pickup_date', 'speed',
        'pickup_longitude_rad', 'pickup_latitude_rad',
        'dropoff_longitude_rad', 'dropoff_latitude_rad'
    ],
    errors="ignore"
)


  return df

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--hidden_layers", nargs="+", type=int, required=True)
    parser.add_argument("--activation", choices=["relu", "sigmoid"], required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--logs_dir", required=True)
    parser.add_argument("--pre_processing_params_dir", required=True)

    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)
    os.makedirs(args.pre_processing_params_dir, exist_ok=True)

    # Load data
    df_train = pd.read_csv(args.train_data)
    df_val = pd.read_csv(args.val_data)

    # Preprocess
    df_train = preprocess_data(df_train)
    df_val = preprocess_data(df_val)

    X_train = df_train.drop(columns=["trip_duration"]).to_numpy(np.float32)
    y_train = np.log1p(df_train["trip_duration"]).to_numpy(np.float32).reshape(-1, 1)

    X_val = df_val.drop(columns=["trip_duration"]).to_numpy(np.float32)
    y_val = np.log1p(df_val["trip_duration"]).to_numpy(np.float32).reshape(-1, 1)

    # Scaling
    feature_mean = df_train.drop(columns=["trip_duration"]).mean()
    feature_std  = df_train.drop(columns=["trip_duration"]).std(ddof=0).replace(0, 1e-8)

    X_train = (X_train - feature_mean.values) / feature_std.values
    X_val   = (X_val   - feature_mean.values) / feature_std.values


    # Train model
    model = MyRegressor(
        n_features=X_train.shape[1],
        hidden_layers=args.hidden_layers,
        activation_function=args.activation,
        learning_rate=args.learning_rate
    )

    logs = model.train(
        X_train, y_train,
        X_val, y_val,
        batch_size=args.batch_size,
        epochs=args.epochs
    )

    sizes = "_".join(map(str, args.hidden_layers))
    base_name = f"hidden_{sizes}_{args.activation}"

    # Save model
    model.save_model(
        os.path.join(args.model_dir, f"model_{base_name}.npz")
    )

    # Save logs
    np.savez(
        os.path.join(args.logs_dir, f"logs_{base_name}.npz"),
        training_loss=logs["training_loss"],
        val_loss=logs["val_loss"],
        total_time=["training_time"]
    )

    # Save preprocessing params
    with open(
    os.path.join(args.pre_processing_params_dir, f"pre_process_{base_name}.json"),
    "w"
    ) as f:
        json.dump({
        "feature_mean": feature_mean.to_dict(),
        "feature_std": feature_std.to_dict(),
        "target_transform": "log1p"
    }, f, indent=4)


if __name__ == "__main__":
    main()
