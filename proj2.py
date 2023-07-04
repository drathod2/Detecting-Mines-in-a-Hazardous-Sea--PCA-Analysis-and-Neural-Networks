############################
# Name: Divyesh Rathod
# ASU ID: 1225916954
# Project - 2
############################


# Import necessary libraries
import numpy as np
import pandas as pd
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from warnings import filterwarnings

# Ignore warning messages
filterwarnings('ignore')

# Load data from CSV file
df = pd.read_csv('sonar_all_data_2.csv')

# Calculate the number of features in the dataset
features = df.shape[1]-2

# Create an empty array to store the PCA results
components_accuracy_table = np.array((features, 2))

# Create empty lists to store the test accuracies, number of components, and column labels
column_label = []
component_number = []
test_accuracy = []

# Create column labels for the dataset
for i in range(features):
    column_label.append('Sample Time: ' + str(i+1))

# Add labels for the target variable
column_label.append('Rock_Mines_int')
column_label.append('Rock_Mines_str')

# Assign the column labels to the dataframe
df.columns = column_label

# Extract the feature variables (X) and target variable (y)
X = df.iloc[:, 0:features]
y = df.iloc[:, features]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=18)

# # Hint to figure out how to interpret the confusion matrix

# rocks = 0                # initialize counters
# mines = 0
# for obj in y_test:    # for all of the objects in the test set
#     if obj == 2:         # mines are class 2, rocks are class 1
#         mines += 1     # increment the appropriate counter
#     else:
#         rocks += 1
# print("rocks",rocks,"   mines",mines)    # print the results


# Define the MLPClassifier model parameters
mlp = MLPClassifier(activation='logistic',
                    max_iter=2000,
                    solver='adam',
                    alpha=0.00001,
                    hidden_layer_sizes=(100),
                    tol=0.0001,
                    random_state=18,
                    )

# Train the model with PCA for different numbers of components
for i in range(1, features+1):
    
    # Apply PCA to the training and testing data
    pca = PCA(n_components=i, random_state=1)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Fit the MLPClassifier model with the PCA training data
    X_train_mlp = mlp.fit(X_train_pca, y_train)

    # Make predictions on the test data
    y_predict = mlp.predict(X_test_pca)

    # Calculate and store the accuracy score of the predictions
    acc_score = round(accuracy_score(y_test, y_predict) * 100, 2)
    test_accuracy.append(acc_score)
    component_number.append(i)
    

    # Print progress message
    sys.stdout.write("\rRunning PCA components: " + str(i))
    sys.stdout.flush()


# Create a dataframe of the PCA results
components_accuracy_table = np.column_stack((component_number,test_accuracy))
df = pd.DataFrame(components_accuracy_table, columns=['PCA Componenets','Test Accuracy'])

# Print the results of the PCA analysis
print("\n\nNumber of PCA components and its test accuracy")
print(df)

# Find the maximum test accuracy and its index
max_test_acc = max(test_accuracy)
max_index = test_accuracy.index(max_test_acc)

# Print the number of components that achieved the maximum accuracy
print("\nNumber of PCA components that achieved maximum accuracy: " +str(max_index+1))

# Print the maximum accuracy achieved by the model
print("\nMaximum accuracy: " + str(max_test_acc))

# Plot the test accuracies against different number of PCA components
plt.plot(component_number, test_accuracy, 'ro-')
plt.xlabel("Number of PCA components")
plt.ylabel("Test accuracy")
plt.title("PCA components VS Test accuracy")
plt.show()

# Print the confusion matrix for the predicted values
print("\nConfusion matrix")
confusion_mat = confusion_matrix(y_test, y_predict)
print(confusion_mat)