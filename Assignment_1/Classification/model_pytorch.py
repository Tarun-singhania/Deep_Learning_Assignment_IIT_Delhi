import torch
import torch.nn as nn

class ForestCoverNet(nn.Module):
    """
    Neural Network for Forest Cover Type Classification
    
    REQUIRED SIGNATURE:
    - __init__(self, input_dim, num_classes=7)
    - forward(self, x) -> returns logits of shape (batch_size, num_classes)
    
    Implement your architecture below.
    """
    
    def __init__(self, input_dim, num_classes=7):
        super(ForestCoverNet, self).__init__()
        
        # TODO: Implement your model architecture
        super(ForestCoverNet, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        # Layers
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        # Activation Function - RELU 
        self.relu = nn.ReLU()

        # He initialization
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        nn.init.kaiming_normal_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # TODO: Implement forward pass
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits