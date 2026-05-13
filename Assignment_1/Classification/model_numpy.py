import numpy as np

class ForestCoverNet:
    """
    Neural Network for Forest Cover Type Classification (NumPy implementation)
    
    REQUIRED SIGNATURE:
    - Structure of NN : [input_dim , 256 , 128 , num_classes] 
    - __init__(self, input_dim, num_classes=7)
    - forward(self, x) -> returns logits of shape (batch_size, num_classes)
    """
    
    def __init__(self, input_dim, num_classes=7):
        self.input_dim=input_dim
        self.num_classes=num_classes

        # He(normal) initilization
        self.W1 = np.random.randn(input_dim,256)*np.sqrt(2.0/input_dim)
        self.b1 = np.zeros((1,256))

        self.W2 = np.random.randn(256,128)*np.sqrt(2.0/256)
        self.b2 = np.zeros((1,128))

        self.W3 = np.random.randn(128,num_classes)*np.sqrt(2.0/128)
        self.b3 = np.zeros((1,num_classes))

    def relu(self,z):
        return np.maximum(0,z)
    
    def forward(self, x):
        """
        Args:
            x: Input array of shape (batch_size, input_dim)
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # TODO: Implement forward pass using NumPy
        self.z1 = x @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.relu(self.z2)

        self.logits = self.a2 @ self.W3 + self.b3

        return self.logits

    def load_state_dict(self, state_dict):
        self.W1 = state_dict["W1"]
        self.b1 = state_dict["b1"]
        self.W2 = state_dict["W2"]
        self.b2 = state_dict["b2"]
        self.W3 = state_dict["W3"]
        self.b3 = state_dict["b3"]
    
    def __call__(self, x):
        return self.forward(x)