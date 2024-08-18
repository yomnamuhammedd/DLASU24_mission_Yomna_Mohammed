import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

x_train_padding = np.load('/Users/yomnaamuhammedd/Desktop/ASU Racing Team Task/X_train_padding.npy')
y_train_padding =np.load('/Users/yomnaamuhammedd/Desktop/ASU Racing Team Task/y_train_padding.npy')


device = 'mps' if torch.backends.mps.is_available() else 'cpu'
if device == 'mps':
    torch.manual_seed(42)

print(f'Using Device {device}')
print(x_train_padding.shape)
print(y_train_padding.shape)

LEARNING_RATE = 0.001
BATCH_SIZE = 4
NUM_EPOCHS = 10
SEQ_LEN = x_train_padding.shape[1]
print("Number of time steps",SEQ_LEN)

def createTensors(x,device):
    x = torch.from_numpy(x).float().to(device)
    # y = torch.tensor(y, dtype=torch.float32)
   # Reshape each  to (samples, features, timesteps) for Conv1d
    x = x.permute(0,2,1)
    # y = y.permute(0,2,1)
    
    # Create tensordataset and and the data laoder with the given batchsize.
    # dataset = TensorDataset(x, y)
    # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return x

# Split Data into train and val
x_train, x_val, y_train, y_val = train_test_split(x_train_padding, y_train_padding,
                                                  test_size=0.04, random_state=42)
                  
x_train = createTensors(x_train,device)
y_train = createTensors(y_train,device)
x_val = createTensors(x_val,device)
y_val = createTensors(y_val,device)


print(f"Shape of training samples: {(x_train.shape)}")
print(f"Shape of validation samples: {(x_val.shape)}")
print(f"Shape of training samples: {(y_train.shape)}")
print(f"Shape of validation samples: {(y_val.shape)}")

# Create CNN Network:
class CNNNetworkSeq(nn.Module):
    def __init__(self, n_in, n_out):
        super(CNNNetworkSeq, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=n_in, out_channels=32, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.pool1 = nn.AvgPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding='same')
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.6)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding='same')
        self.pool3 = nn.MaxPool1d(kernel_size=2)  # Ensure this is a pooling layer
        self.bn3 = nn.BatchNorm1d(128)

        # Calculate the output size after conv layers and pooling
        self._to_linear_input_size = self._get_conv_output_size()

        self.linear1 = nn.Linear(self._to_linear_input_size, 256)
        self.fc = nn.Linear(256, 14 * 12246) 

    def _get_conv_output_size(self):
        n = torch.zeros(1, 12, 12246)  
        x = self.conv1(n)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.relu(x)

        return x.numel()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout1(x)
       
        x = x.view(x.size(0), -1)  # Flatten for the linear layers
        x = self.linear1(x)
        x = self.dropout1(x)
        x = self.fc(x)

        x = x.view(-1, 14, 12246)  # Reshape to match target shape

        return x
    
    def train_model(self, x_train, y_train, loss_fn, optimizer, batch_size):
        self.train()
        num_batches = len(x_train) //batch_size
        training_error = 0.0

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
        
            inputs = x_train[start_idx:end_idx]
            labels = y_train[start_idx:end_idx]
            optimizer.zero_grad()  # Clear previous gradients

            outputs = self(inputs)  # Forward pass

            loss = loss_fn(outputs, labels)
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            training_error += loss.item()

        training_error /= num_batches  
        return training_error
    
    def validate_model(self, x_val, y_val, loss_fn, batch_size):

        self.eval()
        num_batches = len(x_val) // batch_size
        validation_error = 0.0

        with torch.no_grad():
            
            for i in range(num_batches):
    
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
            
                inputs = x_val[start_idx:end_idx]
                labels = y_val[start_idx:end_idx]
            
                outputs = self(inputs)  # Forward pass

                loss = loss_fn(outputs, labels)

                validation_error += loss.item()

        validation_error /= num_batches 
        return validation_error
  
 
model = CNNNetworkSeq(x_train.shape[1],y_train.shape[1]*y_train.shape[-1]).to(device)

loss_fn = nn.MSELoss()
optimizer = optim.Adagrad(model.parameters(), lr=LEARNING_RATE,weight_decay=0.0001)

train_losses = []
val_losses = []

print("----------------------Model Training------------------")
for epoch in range(NUM_EPOCHS):
    train_loss = model.train_model(x_train,y_train, loss_fn, optimizer,BATCH_SIZE)
    val_loss = model.validate_model(x_val,y_val, loss_fn,BATCH_SIZE)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Validation Loss = {val_loss:.6f}")


epochs = range(1, len(train_losses) + 1)
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('CNNNetworkPlot-Sequential.png')
plt.show()
        
