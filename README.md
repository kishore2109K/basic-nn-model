# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

A neural network is a computer program inspired by how our brains work. It's used to solve problems by finding patterns in data. Imagine a network of interconnected virtual "neurons." Each neuron takes in information, processes it, and passes it along.

A Neural Network Regression Model is a type of machine learning algorithm that is designed to predict continuous numeric values based on input data. It utilizes layers of interconnected nodes, or neurons, to learn complex patterns in the data. The architecture typically consists of an input layer, one or more hidden layers with activation functions, and an output layer that produces the regression predictions.
This model can capture intricate relationships within data, making it suitable for tasks such as predicting prices, quantities, or any other continuous numerical outputs.

## Neural Network Model

![image](https://github.com/RoopakCS/basic-nn-model/assets/139228922/ebcf9b27-c04b-4366-b686-7c2eb3c9e940)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Kishore K
### Register Number: 212223040101

## Importing Modules:
```python
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from google.colab import auth
import gspread
from google.auth import default
```

## Authenticate & Create Dataframe using Data in Sheets:
```python
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('MyMLData').sheet1
data = worksheet.get_all_values()

dataset1 = pd.DataFrame(data[1:], columns=data[0])
dataset1 = dataset1.astype({'Input':'float'})
dataset1 = dataset1.astype({'Output':'float'})

dataset1.head()
```

## Assigning input column to X and output column to y:
```python
X = dataset1[['Input']].values
y = dataset1[['Output']].values
```

## Splitting testing and training data:
```python
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
```

## Pre processing:
```python
Scaler = MinMaxScaler()
```

## Scaling the input for training:
```python
Scaler.fit(X_train)
```

## Transforming the scaled input:
```python
X_train1 = Scaler.transform(X_train)
```

## Creating the model:
```python
ai_brain=Sequential([
    Dense(units = 4, activation = 'relu',input_shape = [1]),
    Dense(units = 3, activation = 'relu'),
    Dense(units = 1)
])
```

## Compiling the model:
```python
ai_brain.compile(optimizer='rmsprop',loss='mse')
```

## Fitting the model:
```python
ai_brain.fit(X_train1,y_train,epochs=2000)
```

## Plot the loss:
```python
loss_df = pd.DataFrame(ai_brain.history.history)

loss_df.plot()
```

## Transforming the model:
```python
X_test1 = Scaler.transform(X_test)
```

## Evaluate the model and predicting for some value:
```python
ai_brain.evaluate(X_test1,y_test)

X_n1 = [[30]]

X_n1_1 = Scaler.transform(X_n1)

ai_brain.predict(X_n1_1)
```
## Dataset Information

![MyMLData - Sheet1_page-0001](https://github.com/RoopakCS/basic-nn-model/assets/139228922/85ee31ea-a591-4b67-b775-c92ed6c90c19)

## OUTPUT

### Training Loss Vs Iteration Plot
![download (1)](https://github.com/RoopakCS/basic-nn-model/assets/139228922/68cbaa5f-f929-45b7-b096-84bcb07ee3f2)


### Test Data Root Mean Squared Error

![image](https://github.com/RoopakCS/basic-nn-model/assets/139228922/511146ac-e6c0-4b53-a020-f527d3f4b06b)

### New Sample Data Prediction

![Screenshot 2024-02-25 131830](https://github.com/RoopakCS/basic-nn-model/assets/139228922/4372d9a5-8e7f-4a38-980d-1d10e432a5b2)

## RESULT

A neural network regression model for the given dataset has been developed Sucessfully.
