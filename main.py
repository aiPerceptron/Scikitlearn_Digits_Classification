# when you dont use x_y = True you use a bunch with 7 keys
# when you use it you get a bunch with 2 keys, x and y
# Reasons why I would use return_X_y
# If I just want to not worry about the rest of the data, like the description.
# Reasons why I wouldnt use return_X_y
# To get all of the data.

from sklearn.datasets import load_digits

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

# step one: data
digits = load_digits() #return_X_y=False, as_frame=False
X = digits["data"]  # This is the flattened version of digits["images"]
y = digits["target"]

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.25, shuffle=True)
#print(X)
#print(type(X))
#print(X.shape)

#y = digits["target"]
#print(y)


#plt.matshow(digits["images"][0], cmap="gray")
number = X[0].reshape(8,8) # Here you're unflattening the flattened data
number_image = digits["images"]#[0] 

#plt.matshow(number, cmap="gray")
fig, axs = plt.subplots(ncols=4, nrows=4, figsize=(5.5, 4.5), 
                        layout="constrained")
for row in range(4):
    for col in range(4):
        number = X[row+col].reshape(8,8)
        axs[row, col].matshow(number,cmap="gray") 
        axs[row,col].set_xticks([y[row+col]])
        axs[row,col].set_yticks([])
fig.suptitle('Numbers Dataset')

plt.show()
print(X.shape)
number_image.shape

# step two picking the algorithm

model = MLPClassifier(max_iter=1150) # in one layer 16 neurons works really well
model = GridSearchCV(model, {"hidden_layer_sizes":[(16,),(4,2),(4,4),(8,8),(64,8),(64,1),(8,64)]}, verbose=4, cv=2) # Gridsearch picks the best hidden layer size for the model. 

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# step five: plotting and analysis

accurate_score = round(accuracy_score(y_test, y_pred)*100, 2) 
print(str(accurate_score) + "% accuracy")

#print(y[0])

