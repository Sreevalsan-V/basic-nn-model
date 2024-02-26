# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

A neural network is a computational model inspired by the structure and function of the human brain. It is a type of machine learning algorithm that processes information through interconnected nodes, known as neurons or artificial neurons. These neurons are organized into layers: an input layer, one or more hidden layers, and an output layer.
In a neural network, each connection between neurons has an associated weight, and the network learns by adjusting these weights based on input data and desired output.

The model below has one input layer,The input layer receives the initial data or features for the neural network to process. Each node in this layer represents a specific feature of the input data.

There are 5 hidden layers in this model, making it a deep neural network. These hidden layers are responsible for learning complex patterns and representations from the input data. Each hidden layer consists of nodes, and the connections between these nodes have associated weights. These weights are adjusted during the learning process to enhance the network's ability to capture and generalize from the input data.

The output layer produces the final result or prediction. Each node in the output layer represents either a class (for classification tasks) or a continuous value (for regression tasks). The exact nature of the output depends on the specific requirements of the problem being addressed.


## Neural Network Model

![alt text](image.png)

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

## PROGRAM: Basic neural regression model
### Name: SREEVALSAN V
### Register Number: 212223240158

## Importing Modules
```
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential as Seq
from tensorflow.keras.layers import Dense as Den
from tensorflow.keras.metrics import RootMeanSquaredError as rmse
```
## Authenticate & Create Dataframe using Data in Sheets
```
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
sheet = gc.open('Data1').sheet1
rows = sheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'INPUT':'int'})
df = df.astype({'OUTPUT':'int'})
df.head(22)
```
## Assign X and Y values
```
X = df[["INPUT"]] .values
y = df[["OUTPUT"]].values
```

## Normalize the values & Split the data
```
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)
```
## Create a Neural Network & Train it
```
AI_Brain = Sequential([
    Dense(units = 5, activation = 'relu', input_shape=[1]),
    Dense(units = 3, activation = 'relu'),
    Dense(units = 1)
])

AI_Brain.compile(optimizer= 'rmsprop', loss="mse")
AI_Brain.fit(X_train1,y_train,epochs=5000)
AI_Brain.summary()
```
## Plot the Loss
```
loss_df = pd.DataFrame(AI_Brain.history.history)
loss_df.plot()
```
## Evaluate the model

```
X_test1 = Scaler.transform(X_test)
AI_Brain.evaluate(X_test1,y_test)
```
## Predict for some value

```
X_n1 = [[20]]
X_n1_1 = Scaler.transform(X_n1)
AI_Brain.predict(X_n1_1)
```

## Dataset Information

![alt text](image-1.png)

## OUTPUT

### Training Loss Vs Iteration Plot

![alt text](image-2.png)

### Final Loss

![alt text](image-4.png)

### New Sample Data Prediction

![alt text](image-3.png)

## RESULT

Thus the Process of developing a neural network regression model for the created dataset is successfully executed.
