
# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Loading the dataset
df = pd.read_csv('/content/Social_Network_Ads.csv')

# Exploring the dataset
df.head(10)

df.info()

df.describe()

# Selecting the features and the target variable
X = df[["Age", "EstimatedSalary"]] # Features
y = df["Purchased"] # Target variable

# Splitting the dataset into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Scaling the features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Creating and fitting the logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predicting the test set results
y_pred_log = log_reg.predict(X_test)

# Evaluating the logistic regression model
acc_log = accuracy_score(y_test, y_pred_log)
cm_log = confusion_matrix(y_test, y_pred_log)
cr_log = classification_report(y_test, y_pred_log)

# Printing the evaluation metrics
print("Accuracy of logistic regression model:", acc_log)

import seaborn as sns
ax = sns.heatmap(cm_log, annot=True, cmap='Blues')
ax.set_title('Confusion Matrix for logistic regression model \n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
## Display the visualization of the Confusion Matrix.
plt.show()

print("Classification report of logistic regression model:\n", cr_log)

plt.figure(figsize=(10,6))
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, log_reg.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Train set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Plotting the decision boundary for the logistic regression model on the test set
plt.figure(figsize=(10,6))
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, log_reg.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Creating and fitting the SVM model
svm = SVC(probability=True)
svm.fit(X_train, y_train)

# Predicting the test set results
y_pred = svm.predict(X_test)
y_prob = svm.predict_proba(X_test)

# Evaluating the SVM model
acc_svm = accuracy_score(y_test, y_pred)
cm_svm = confusion_matrix(y_test, y_pred)
cr_svm = classification_report(y_test, y_pred)

# Printing the evaluation metrics
print("Accuracy of SVM model:", acc_svm)

import seaborn as sns
ax = sns.heatmap(cm_svm, annot=True, cmap='Blues')
ax.set_title('Confusion Matrix for SVM model \n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
## Display the visualization of the Confusion Matrix.
plt.show()

print("Classification report of SVM model:\n", cr_svm)

# Plotting the decision boundary for the SVM model on the train set
plt.figure(figsize=(10,6))
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, svm.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Train set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Plotting the decision boundary for the SVM model on the test set
plt.figure(figsize=(10,6))
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, svm.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Comparing the models
if acc_log > acc_svm:
  print("Logistic regression model is better than SVM model.")
elif acc_log < acc_svm:
  print("SVM model is better than logistic regression model.")
else:
  print("Both models have the same accuracy.")

# Visualizing the results
# Plotting the accuracy scores of the models on a bar chart
plt.figure(figsize=(10,6))
models = ["Logistic Regression", "SVM"]
accuracy = [acc_log, acc_svm]
plt.bar(models, accuracy, color=["blue", "orange"])
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison of Logistic Regression and SVM Models")
plt.show()

from sklearn.ensemble import VotingClassifier

# Creating and fitting the VotingClassifier with SVM and logistic regression models
voting_clf = VotingClassifier(estimators=[('svm', svm), ('log_reg', log_reg)], voting='soft')
voting_clf.fit(X_train, y_train)

# Predicting the test set results
y_pred_voting = voting_clf.predict(X_test)

# Evaluating the VotingClassifier model
acc_voting = accuracy_score(y_test, y_pred_voting)
cm_voting = confusion_matrix(y_test, y_pred_voting)
cr_voting = classification_report(y_test, y_pred_voting)

# Printing the evaluation metrics
print("Accuracy of VotingClassifier model:", acc_voting)

print("Classification report of VotingClassifier model:\n", cr_voting)

import seaborn as sns
ax = sns.heatmap(cm_voting, annot=True, cmap='Blues')
ax.set_title('Confusion Matrix for VotingClassifier model \n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
## Display the visualization of the Confusion Matrix.
plt.show()

# Comparing the models
if acc_voting > max(acc_log, acc_svm):
  print("VotingClassifier model is better than both SVM and logistic regression models.")
elif acc_voting < min(acc_log, acc_svm):
  print("VotingClassifier model is worse than both SVM and logistic regression models.")
else:
  print("VotingClassifier model is in between SVM and logistic regression models.")

# Plotting the class probabilities of the voting classifier and the individual classifiers for a given sample
sample = X_test[0] # You can change this to any sample you want
y_true = y_test[0] # The true label of the sample
y_pred_svm = svm.predict([sample]) # The predicted label by SVM
y_pred_log = log_reg.predict([sample]) # The predicted label by logistic regression
y_pred_voting = voting_clf.predict([sample]) # The predicted label by voting classifier
y_prob_svm = svm.predict_proba([sample]) # The predicted probabilities by SVM
y_prob_log = log_reg.predict_proba([sample]) # The predicted probabilities by logistic regression
y_prob_voting = voting_clf.predict_proba([sample]) # The predicted probabilities by voting classifier
print("The sample is:", sample)
print("The true label is:", y_true)
print("The predicted label by SVM is:", y_pred_svm)
print("The predicted label by logistic regression is:", y_pred_log)
print("The predicted label by voting classifier is:", y_pred_voting)
plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.bar(['0', '1'], y_prob_svm[0], color=['red', 'green'])
plt.title('SVM')
plt.xlabel('Class')
plt.ylabel('Probability')
plt.subplot(2,2,2)
plt.bar(['0', '1'], y_prob_log[0], color=['red', 'green'])
plt.title('Logistic Regression')
plt.xlabel('Class')
plt.ylabel('Probability')
plt.subplot(2,2,3)
plt.bar(['0', '1'], y_prob_voting[0], color=['red', 'green'])
plt.title('Voting Classifier')
plt.xlabel('Class')
plt.ylabel('Probability')
plt.show()

# Plotting the accuracy scores of the voting classifier and the individual classifiers on the test set
plt.figure(figsize=(10,6))
models = ["SVM", "Logistic Regression", "Voting Classifier"]
accuracy = [acc_svm, acc_log, acc_voting]
plt.bar(models, accuracy, color=["blue", "orange", "purple"])
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison of Different Models")
plt.show()