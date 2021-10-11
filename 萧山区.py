# Import necessary packages
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math

import sklearn
from sklearn import preprocessing

import matplotlib.pyplot as plt

import pyneurgen
from pyneurgen.neuralnet import NeuralNet
from pyneurgen.recurrent import NARXRecurrent

import codecs

# Read in the data set
df = pd.read_csv('萧山区.csv', encoding='utf-8')

# Construct the feature set and the target set
X = df.iloc[:, 0:15]
y = df.iloc[:, 15:]

scX = preprocessing.MinMaxScaler(feature_range=(0, 1))
X = np.array(X).reshape((len(X), 15))
X = scX.fit_transform(X)
X = X.tolist()

scy = preprocessing.MinMaxScaler(feature_range=(0, 1))
y = np.array(y).reshape((len(y), 3))
y = scy.fit_transform(y)
y = y.tolist()

# Define the hyper-parameters of the model
input_nodes = 15
hidden_nodes = 10
output_nodes = 3
output_order = 3
input_order = 15

incoming_weight_from_output = 0.1
incoming_weight_from_input = 0.8

# Define the model parameters
model_NARX = NeuralNet()
model_NARX.init_layers(input_nodes, [hidden_nodes], output_nodes,
                       NARXRecurrent(output_order,
                                     incoming_weight_from_output,
                                     input_order,
                                     incoming_weight_from_input))

model_NARX.randomize_network()
model_NARX.layers[1].set_activation_type('sigmoid')
model_NARX.set_learnrate(0.35)
model_NARX.set_all_inputs(X)
model_NARX.set_all_targets(y)

# Set up and train the model
model_NARX.set_learn_range(0, len(X) - 1)
model_NARX.set_test_range(0, len(X) - 1)
model_NARX.learn(epochs=50, show_epoch_results=True, random_testing=False)

mse = model_NARX.test()

print('Approximate MSE for test set =', mse)

# Visualize the result
target = np.array([item[0][0] for item in model_NARX.test_targets_activations])
predicted = [item[1] for item in model_NARX.test_targets_activations]

predicted = list(predicted)

kindergarten = []
for i in range(len(y)):
    kindergarten.append(y[i][0])

predicted_kindergarten = []
for i in range(len(predicted)):
    predicted_kindergarten.append(predicted[i][0])

primary_school = []
for i in range(len(y)):
    primary_school.append(y[i][1])

predicted_primary_school = []
for i in range(len(predicted)):
    predicted_primary_school.append(predicted[i][1])

junior_high_school = []
for i in range(len(y)):
    junior_high_school.append(y[i][2])

predicted_junior_high_school = []
for i in range(len(predicted)):
    predicted_junior_high_school.append(predicted[i][2])

fig = plt.figure(figsize=(6, 9))

ax1 = fig.add_subplot(3, 1, 1)
ax1.plot(kindergarten, color='blue')
ax1.plot(predicted_kindergarten, color='red')
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.xlabel("Year (2007 - 2019)", fontsize=7)
plt.title("Actual versus Predicted Kindergarten Enrollments", fontsize=7)
plt.legend(['Kindergarten Enrollment', 'Predicted Kindergarten Enrollment'], prop={'size': 7})

ax2 = fig.add_subplot(3, 1, 2)
ax2.plot(primary_school, color='blue')
ax2.plot(predicted_primary_school, color='red')
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.xlabel("Year (2007 - 2019)", fontsize=7)
plt.title("Actual versus Predicted Primary School Enrollments", fontsize=7)
plt.legend(['Primary School Enrollment', 'Predicted Primary School Enrollment'], prop={'size': 7})

ax3 = fig.add_subplot(3, 1, 3)
ax3.plot(junior_high_school, color='blue')
ax3.plot(predicted_junior_high_school, color='red')
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.xlabel("Year (2007 - 2019)", fontsize=7)
plt.title("Actual versus Predicted Junior High School Enrollments", fontsize=7)
plt.legend(['Junior High School Enrollment', 'Predicted Junior High School Enrollment'], prop={'size': 7})

fig.tight_layout(pad=2.0)
plt.show()