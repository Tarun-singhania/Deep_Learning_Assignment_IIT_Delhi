import numpy as np
import time

class MyRegressor:

  def __init__(self,n_features,hidden_layers,activation_function="sigmoid",learning_rate=0.001,verbose=False):
    self.lr=learning_rate
    self.verbose=verbose         # It is used for that Do you want to printing all epochs result or not?
    self.activation_function=activation_function.lower()

    # It said about how many neurons are present in each layer-- [i/p .....hidden_layers..... o/p]
    self.layers = [n_features]+hidden_layers+[1]

    ''' Step 1 : Network initialization for arbitrary depth and width
    Weight Initilization : It happens automatically when you create object of this class  -->'''
    # we are using Xavier Uniform and He Uniform weight initilization technique for 'sigmoid' and 'relu' respectively
    self.Weights=[]
    self.biases=[]

    if self.activation_function=="sigmoid":
      for i in range(len(self.layers)-1):
        fan_in = self.layers[i]
        fan_out = self.layers[i+1]

        # Last layer (linear output layer)
        if i == len(self.layers) - 2:
          W = np.random.randn(fan_in, fan_out) * 0.01
          b = np.zeros((1, fan_out))
        else:
          limit = np.sqrt(6 / (fan_in + fan_out))
          W = np.random.uniform(-limit, limit, size=(fan_in, fan_out))
          b = np.zeros((1, fan_out))

        self.Weights.append(W)
        self.biases.append(b)
    else:
      for i in range(len(self.layers)-1):
        fan_in = self.layers[i]
        fan_out = self.layers[i+1]

        # Last layer (linear output layer)
        if i == len(self.layers) - 2:
          W = np.random.randn(fan_in, fan_out) * 0.01
          b = np.zeros((1, fan_out))
        else:
          limit = np.sqrt(6/fan_in)
          W = np.random.uniform(-limit, limit, size=(fan_in, fan_out))
          b = np.zeros((1, fan_out))

        self.Weights.append(W)
        self.biases.append(b)

  ''' Step 2 : forward and backward propagation
  (i). Forward Propagation '''
  # Create Sigmoid Function :
  def sigmoid(self, z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


  # Create Relu Function
  def relu(self,z):
    return np.maximum(0, z)

  # Forward Propogation :
  def forward_propagation(self, X):
      activations_results = [X]
      Z_results = []

      for layer_no in range(len(self.Weights)):
          z = activations_results[-1] @ self.Weights[layer_no] + self.biases[layer_no]
          Z_results.append(z)

          # For Output layer: linear activation
          if layer_no == len(self.Weights) - 1:
            a = z
          else:
            if self.activation_function=="sigmoid":
              a = self.sigmoid(z)
            else:
              a = self.relu(z)

          activations_results.append(a)

      return activations_results, Z_results

  '''(ii). BackWard Propagation '''
  # Mean Squared Error loss
  def mse_loss(self, y, y_pred):
      return np.mean((y - y_pred) ** 2)

  # Sigmoid derivative
  def sigmoid_derivative(self, a):
      return a * (1 - a)

  def relu_derivative(self,z):
      z = np.array(z)
      return (z > 0).astype(float)

  def backward_propagation(self, X, y, activations, zs):
      m = X.shape[0]

      if y.ndim == 1:
          y = y.reshape(-1, 1)

      # Gradients
      derv_W = [None] * len(self.Weights)
      derv_b = [None] * len(self.biases)

      # Output layer gradient
      delta = 2 * ((activations[-1] - y) / m)  # dL/dz for linear output

      for i in reversed(range(len(self.Weights))):
          derv_W[i] = activations[i].T @ delta
          derv_b[i] = np.sum(delta, axis=0, keepdims=True)

          if i > 0:
            if self.activation_function=="sigmoid":
              delta = (delta @ self.Weights[i].T) * self.sigmoid_derivative(activations[i])
            else:
              delta = (delta @ self.Weights[i].T) * self.relu_derivative(zs[i-1])

      # Update parameters
      clip_value = 5.0

      for i in range(len(self.Weights)):
          derv_W[i] = np.clip(derv_W[i], -clip_value, clip_value)
          derv_b[i] = np.clip(derv_b[i], -clip_value, clip_value)

          self.Weights[i] -= self.lr * derv_W[i]
          self.biases[i]  -= self.lr * derv_b[i]

  def train(self, X_train, y_train,X_val,y_val,batch_size=32, epochs=300,patience=20):

      if X_train.ndim != 2:
          raise ValueError("X must be 2D (samples, features)")

      if X_train.shape[1] != self.layers[0]:
          raise ValueError(
              f"Expected {self.layers[0]} features, got {X_train.shape[1]}"
          )

      if y_train.ndim == 1:
          y_train = y_train.reshape(-1, 1)
      if y_val.ndim == 1:
          y_val = y_val.reshape(-1, 1)

      n=X_train.shape[0]

      # Uses for Early Stooping :
      max_val_loss = float("inf")
      patience_count = 0

      # Logs dictionary 
      logs = { "training_loss": [], 
              "val_loss": [], 
              "epochs": [],
              "total_time":[] }

      # Time Start
      start_time = time.time()

      for epoch in range(epochs):
          indices = np.random.permutation(n)      # re_suffling the row no's

          for j in range(0, n, batch_size):
            batch_idx = indices[j:j+batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]

            activations, zs = self.forward_propagation(X_batch)
            self.backward_propagation(X_batch, y_batch, activations, zs)

          y_pred_train = self.forward_propagation(X_train)[0][-1]
          train_loss = self.mse_loss(y_train, y_pred_train)

          y_pred_val = self.forward_propagation(X_val)[0][-1]
          val_loss = self.mse_loss(y_val,y_pred_val)


          # Save logs 
          logs["training_loss"].append(train_loss) 
          logs["val_loss"].append(val_loss) 
          logs["epochs"].append(epoch + 1)
          
          if self.verbose:
              print(f"Epoch {epoch+1} : train_Loss: {train_loss:.4f} , val_Loss: {val_loss:.4f}")

          if val_loss < max_val_loss:
                max_val_loss = val_loss
                patience_count = 0
          else:
              patience_count += 1

          if patience_count >= patience:
              print(f"Early stopping at epoch {epoch}")
              break
      total_time = time.time() - start_time
      logs["total_time"] = (total_time)
      
      return logs

  #Prediction
  def predict(self, X):
    return self.forward_propagation(X)[0][-1]
  
  #Save Model
  def save_model(self, path):
        data = {}
        for i, (W, b) in enumerate(zip(self.Weights, self.biases)):
            data[f"W{i}"] = W
            data[f"b{i}"] = b
        np.savez(path, **data)

  #Load model
  def load_model(self, path):
    data = np.load(path)
    self.Weights, self.biases = [], []
    i = 0
    while f"W{i}" in data:
      self.Weights.append(data[f"W{i}"])
      self.biases.append(data[f"b{i}"])
      i += 1