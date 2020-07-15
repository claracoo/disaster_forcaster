from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, AveragePooling2D
from keras.models import Sequential
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
from random import randint
from random import random
import numpy as np
import json

alarmWorthy = 1


class BlackBox:
    # Currently this is similar to heating something in a vacuum.

    def __init__(self, mass, specific_heat, starting_temp, alarm_value):
        # Initial conditions.
        self.mass = mass
        self.specific_heat = specific_heat
        self.starting_temp = starting_temp
        self.alarm_value = alarm_value

    def operation(self, time, power):
        self.operation = "heat"
        self.time = time
        self.power = power

        if self.black_box() > self.alarm_value:
            self.alarm = True
            print(self.black_box())
        else:
            self.alarm = False

        self.output_record()

    def black_box(self):
        slope = self.power / (self.specific_heat * self.mass)
        final_temp = slope * self.time + self.starting_temp
        return final_temp

    def output_record(self):
        print(self.time, self.power,
              self.mass, self.specific_heat, self.starting_temp,
              self.alarm_value, self.alarm)
        output = np.array([np.array([self.time, self.power,
                                     self.mass, self.specific_heat, self.starting_temp]), np.array([self.alarm])])
        return output


TRAINING = 5
TESTING = 1
DATA_POINTS_TRAINING = 6
DATA_POINTS_TESTING = 1


x_train = np.array([])
x_test = np.array([])
y_train = np.array([])
y_test = np.array([])
for i in range(7):
    system = BlackBox(random(), random(), random(), alarmWorthy)
    system.operation(random(), random())
    output = system.output_record()
    if i < 6:
        x_train = np.append(x_train, output[0], axis=0)
        y_train = np.append(y_train, output[1], axis=0)
        # x_train.append(output[0])
        # y_train.append(output[1])
    else:
        x_test = np.append(x_test, output[0], axis=0)
        y_test = np.append(y_test, output[1], axis=0)
        # x_test.append(output[0])
        # y_test.append(output[1])

# x_train_shaped = x_train.reshape(DATA_POINTS_TRAINING, TRAINING)
# y_train_shaped = y_train.reshape(DATA_POINTS_TESTING, TRAINING)
# x_test_shaped = x_test.reshape(DATA_POINTS_TRAINING, TESTING)
# y_test_shaped = y_test.reshape(DATA_POINTS_TESTING, TESTING)

x_train = x_train.reshape(1, 30)
y_train = y_train.reshape(1, 6)
x_test = x_test.reshape(1, 5)
y_test = y_test.reshape(1, 1)


print(x_train)
print(x_test)
print(y_train)
print(y_test)


TRAIN_SET_LIMIT = 1000
TRAIN_SET_COUNT = 100

TRAIN_INPUT = list()
TRAIN_OUTPUT = list()

x_train = list()
x_test = list()
y_train = list()
y_test = list()
for i in range(700):
    system = BlackBox(random(), random(), random(), alarmWorthy)
    system.operation(random(), random())
    output = system.output_record()
    if i < 600:
        x_train.append(output[0])
        y_train.append(output[1])
        # x_train.append(output[0])
        # y_train.append(output[1])
    else:
        x_test.append(output[0])
        y_test.append(output[1])
        # x_test.append(output[0])
        # y_test.append(output[1])


def generateTruesFalses(indexX, indexY):

    truesX = []
    truesY = []
    falsesX = []
    falsesY = []

    for i in range(len(x_test)):
        if y_test[i][0]:
            truesX.append(x_test[i][indexX])
            truesY.append(x_test[i][indexY])
        else:
            falsesX.append(x_test[i][indexX])
            falsesY.append(x_test[i][indexY])
    return [truesX, truesY, falsesX, falsesY]


def generateJSON(outcome):
    solution = []
    fullDict = {}
    time = {}
    power = {}
    mass = {}
    specHeat = {}
    startTemp = {}
    inputData = []
    time["power"] = generateTruesFalses(0, 1)
    time["mass"] = generateTruesFalses(0, 2)
    time["specHeat"] = generateTruesFalses(0, 3)
    time["startTemp"] = generateTruesFalses(0, 4)
    fullDict["time"] = time
    power["time"] = generateTruesFalses(1, 0)
    power["mass"] = generateTruesFalses(1, 2)
    power["specHeat"] = generateTruesFalses(1, 3)
    power["startTemp"] = generateTruesFalses(1, 4)
    fullDict["power"] = power
    mass["time"] = generateTruesFalses(2, 0)
    mass["power"] = generateTruesFalses(2, 1)
    mass["specHeat"] = generateTruesFalses(2, 3)
    mass["startTemp"] = generateTruesFalses(2, 4)
    fullDict["mass"] = mass
    specHeat["time"] = generateTruesFalses(3, 0)
    specHeat["power"] = generateTruesFalses(3, 1)
    specHeat["mass"] = generateTruesFalses(3, 2)
    specHeat["startTemp"] = generateTruesFalses(3, 4)
    fullDict["specHeat"] = specHeat
    startTemp["time"] = generateTruesFalses(4, 0)
    startTemp["power"] = generateTruesFalses(4, 1)
    startTemp["mass"] = generateTruesFalses(4, 2)
    startTemp["specHeat"] = generateTruesFalses(4, 3)
    fullDict["startTemp"] = startTemp
    solution = [fullDict, outcome, inputData]
    # Serializing json
    json_object = json.dumps(solution)
    # Writing to sample.json
    with open("sample.json", "w") as outfile:
        outfile.write(json_object)

# colors = ['red', 'green', "black"]

# fig, ax = plt.subplots()

# # for color in colors:
#     # if colorCount < 1:
#         # n = 38
# tx, ty = truesX, truesY
# fx, fy = falsesX, falsesY
# # extraX, extraY = [0.73427357, 1, 0],  [0.61877181, 1, 0]
# plt.scatter(tx, ty, color=colors[0], alpha=.8, label="True")
# plt.scatter(fx, fy, color=colors[1], alpha=.8, label="False")
# # plt.scatter(extraX, extraY, color=colors[2], alpha=.8, label="Input Data")
# plt.legend(loc='best', shadow=False, scatterpoints=1)
# plt.title("Time vs. Power in likelihood the Beamline will shut down")
# ax.set_ylabel("Power")
# ax.set_xlabel("Time")


# (x_train, y_train), (x_test, y_test) = data
# Importing the required Keras modules containing model and layers
arrx = numpy.array(x_train)
arry = numpy.array(y_train)
arrtestx = numpy.array(x_test)
arrtesty = numpy.array(y_test)
print(arrtestx.shape)
print(arrtesty.shape)
print(arry.shape)
print(arrx.shape)
arrx = arrx.reshape(600, 5, 1)
arrtestx = arrtestx.reshape(100, 5, 1)
# arry = arry.reshape(6, 1, 1)
print(arrx.flatten())

# Creating a Sequential Model and adding the layers
model = Sequential()
# model.add(Conv2D(64, kernel_size=(3,3), ))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
# final layer must have two bc we have 2 final options
model.add(Dense(2, activation=tf.nn.softmax))
# print(type(x_train))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x=arrx, y=arry, epochs=5)

print(model.evaluate(arrtestx, arrtesty))


def predict(inputData):
    if len(inputData) != 5:
        print("Warning, incomplete feature data list.")
        return None
    # X_NEW = np.array([.2, .15, .0621759, .81367, .2377854])
    X_NEW = inputData.reshape(1, 5, 1)
    print(X_NEW.shape)
    outcome = model.predict(X_NEW)
    # print(outcome[0][0], outcome[0][1])

    if outcome[0][1] >= alarmWorthy:
        print("The lab will blow up: ", True)
        return True
    else:
        print("The lab will blow up: ", False)
        return False


# make json file
generateJSON(None)
# Opening JSON file
inputData = []
with open('sample.json', 'r') as openfile:

    # Reading from json file
    json_object = json.load(openfile)
    inputData = json_object[2]
    # print(json_object)

generateJSON(predict(inputData))
