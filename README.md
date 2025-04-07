# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

![ex2 nn model](https://github.com/user-attachments/assets/fe7ea189-ad13-4753-9c96-3a06b43a0674)


## DESIGN STEPS
### STEP 1:
Understand the classification task and identify input and output variables.

# STEP 2:
Gather data, clean it, handle missing values, and split it into training and test sets.

# STEP 3:
Normalize/standardize features, encode categorical labels, and reshape data if needed.

# STEP 4:
Choose the number of layers, neurons, and activation functions for your neural network.

# STEP 5:
Select a loss function (e.g., binary cross-entropy), optimizer (e.g., Adam), and metrics (e.g., accuracy).

# STEP 6:
Feed training data into the model, run multiple epochs, and monitor the loss and accuracy.

# STEP 7:
Save the trained model, export it if needed, and deploy it for real-world use.

## PROGRAM

### Name: Priyadharshini.P
### Register Number:212223240128

```python
# Define Neural Network(Model1)
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 4)
    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=self.fc4(x)
        return x
```
```python
# Initialize the Model, Loss Function, and Optimizer
model =PeopleClassifier(input_size=X_train.shape[1])
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)

```
```python
def train_model(model,train_loader,criterion,optimizer,epochs):
  for epoch in range(epochs):
    model.train()
    for X_batch,y_batch in train_loader:
      optimizer.zero_grad()
      outputs=model(X_batch)
      loss=criterion(outputs,y_batch)
      loss.backward()
      optimizer.step()

  if(epoch+1)%10==0:
    print(f'Epoch [{epoch+1}/{epochs}],Loss:{loss.item():.4f}')
```

## Dataset Information

![image](https://github.com/user-attachments/assets/18bee10f-1b3a-4985-bf72-5c2e02415366)

## OUTPUT

### Confusion Matrix
![image](https://github.com/user-attachments/assets/8a6edd95-9d89-4da8-8b9e-cf87d078dea7)

### Classification Report

![image](https://github.com/user-attachments/assets/0e4481fa-7bc3-4b5f-98aa-0bd325a18e37)


### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/aba8172e-75c4-4239-b0ad-31ce2ae2878e)

## RESULT
Thus a neural network classification model for the given dataset is executed successfully.
