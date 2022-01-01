import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from tensorflow import keras
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from imblearn.over_sampling import SMOTE
import math

# Display Options
pd.set_option('display.max_rows', None)
pd.set_option('expand_frame_repr', False)

# Read Data
data = pd.read_csv("creditcard.csv")

# Data Exploration
print("Data Shape")
print(data.shape, '\n')

print("Data Description")
print(data.describe(), '\n')

print("Check Missing Values")
print(data.isna().any(), '\n')

# Graph of target variable
data.Class.value_counts().plot(kind='bar')
plt.ylabel("Values")
plt.xlabel("Class")
plt.title("Class Distribution of the Dataset - Original")
plt.show()

# Drop Time Variable as it seems unnecessary
data = data.drop(['Time'], axis=1)
X = data.drop('Class', 1)
Y = data.Class

# Split Dataset into testing and training
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=0, stratify=Y)

# Use SMOTE technique to Over-Sample
oversample = SMOTE()
X, Y = oversample.fit_resample(x_train, y_train)

# Plot Over-sampled data
Y.value_counts().plot(kind='bar')
plt.ylabel("Values")
plt.xlabel("Class")
plt.title("Class Distribution of the Dataset - SMOTE Over-Sample")
plt.show()

# Check new shape of over sampled data
print(X.shape)
print(Y.shape)
print(x_test.shape)
print(y_test.shape)

# Gaussian Model
model = GaussianNB().fit(X, Y)
tn, fp, fn, tp = confusion_matrix(y_test, model.predict(x_test)).ravel()
print("Parameters NB")
print("True Negatives:", tn)
print("False Positives", fp)
print("False Negatives:", fn)
print("True Positives:", tp, '\n')
print("Precision:", round(tp / (tp + fp), 3))
print("F1 Score", round(f1_score(y_test, model.predict(x_test)), 3))
print("Recall:", round(tp / (tp + fn), 3), '\n')

print("Confusion Matrix")
print(confusion_matrix(y_test, model.predict(x_test)), '\n')
plot_confusion_matrix(model, x_test, y_test)
plt.title("Confusion Matrix NB")
plt.show()

# Model Decision Tree
model = DecisionTreeClassifier(random_state=0).fit(X, Y)

tn, fp, fn, tp = confusion_matrix(y_test, model.predict(x_test)).ravel()
print("Parameters Decision Tree")
print("True Negatives:", tn)
print("False Positives", fp)
print("False Negatives:", fn)
print("True Positives:", tp, '\n')
print("Precision:", round(tp / (tp + fp), 3))
print("F1 Score", round(f1_score(y_test, model.predict(x_test)), 3))
print("Recall:", round(tp / (tp + fn), 3), '\n')

print("Confusion Matrix")
print(confusion_matrix(y_test, model.predict(x_test)), '\n')
plot_confusion_matrix(model, x_test, y_test)
plt.title("Confusion Matrix Decision Tree")
plt.show()

# Model Random Forest with best parameters
error_rate = []
parameters = []
for i in range(1, 11):
    for j in range(1, 6):
        model = RandomForestClassifier(n_estimators=i, max_depth=j, criterion="entropy", random_state=0).fit(X, Y)
        error_rate.append(1 - accuracy_score(y_test, model.predict(x_test)))
        parameters.append([i, j])

data = pd.DataFrame({"Parameters": parameters, "Error_Rate": error_rate})
data = data.sort_values(['Error_Rate'])

optimal_parameters = data.loc[data['Error_Rate'] == data['Error_Rate'].min(), 'Parameters']
optimal_trees = list(optimal_parameters)[0][0]
optimal_depth = list(optimal_parameters)[0][1]

model = RandomForestClassifier(n_estimators=optimal_trees, max_depth=optimal_depth, criterion="entropy",
                               random_state=0).fit(X, Y)

tn, fp, fn, tp = confusion_matrix(y_test, model.predict(x_test)).ravel()
print("Parameters Random Forest")
print("True Negatives:", tn)
print("False Positives", fp)
print("False Negatives:", fn)
print("True Positives:", tp, '\n')
print("Precision:", round(tp / (tp + fp), 3))
print("F1 Score", round(f1_score(y_test, model.predict(x_test)), 3))
print("Recall:", round(tp / (tp + fn), 3), '\n')

print("Confusion Matrix")
print(confusion_matrix(y_test, model.predict(x_test)), '\n')
plot_confusion_matrix(model, x_test, y_test)
plt.title("Confusion Matrix Random Forest")
plt.show()

# Model Neural Network


def scheduler(epoch, learning_rate):
    if epoch < 10:
        return learning_rate
    else:
        return learning_rate * math.exp(-0.1)


model = Sequential()
model.add(keras.layers.Dense(29, activation="relu", input_shape=(29,)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(20, activation="relu"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(10, activation="relu"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(1, activation="sigmoid"))

custom_early_stopping = EarlyStopping(
    monitor='val_precision',
    patience=10,
    restore_best_weights=True)

opt = SGD(learning_rate=0.01,
          momentum=0.9,
          nesterov=True)

adam = Adam(
    lr=0.0005,
    beta_1=0.9,
    beta_2=0.999,
    decay=0.0)

model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["Precision", "Recall"])
history = model.fit(X, Y, validation_split=0.05, epochs=50,
                    callbacks=[custom_early_stopping, LearningRateScheduler(scheduler)])

score = model.evaluate(x_test, y_test)
print(model.metrics_names)
print(score)
f1_Score = 2 * ((score[1] * score[2]) / (score[1] + score[2]))
print("F1 score", f1_Score)
