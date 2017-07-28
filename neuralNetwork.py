# This is a simple feedforward neural network programm i created by using
# the book "Make your own Neural Network" by Tariq Rashids. It can read in
# pictures of handwritten digits from 0 to 9 of an 28x28 PNG-Image. It's using
# a 3 layer network and the sigmoid activiation function. For Training i used
# the MNIST-Dataset in a csv-format of 66000 examples. I tested it on several
# learning rates and epoches aswell as 100 and 200 hiddennodes. My best rate of
# success was 94,83% which is good enough for this simple network, even though
# there way better ones out there.
#
# All Datasets and a table with my test results can be found here:
# https://github.com/Tobi96/nn-in-python


# imports
import numpy
import scipy.special

# simple feedforward neural network class
class neuralNetwork:

    # initialize network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set layers of the NN
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # set learningrate
        self.lr = learningrate

        # set weight matrices: weights input to hidden and weights hidden to output
        # using a normal distribution in form of 1/sqrt(number of incomming connections)
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0,5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0,5), (self.onodes, self.hnodes))

        #activiation function is the sigmoid function (expit())
        self.activation_function = lambda x: scipy.special.expit(x)

    # train network
    def train(self, inputs_list, targets_list):
        # convert inputs to 2d Arrays
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layers
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layers
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # Backpropagation
        # error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer errors are the output_errors split by weights
        hidden_errors = numpy.dot(self.who.T, output_errors)
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

    # query network
    def query(self, inputs_list):
        # convert inputs_list to 2d Array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layers
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layers
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

# main
# Configuration
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.2
epochs = 9 # epochs is the number of times the network uses the training set to train

# create a instance of NN
nn = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# load mnist dataset
training_data_file = open("mnist_train.csv",'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# train the network
for e in range(epochs):
    for record in training_data_list:
        # split data
        all_values = record.split(',')
        # put them in a array and normalise them
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create output arrays with 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        # set 0.99 at goal number
        targets[int(all_values[0])] = 0.99
        # let it train on this data
        nn.train(inputs,targets)

# test
# suppress scientific notation
numpy.set_printoptions(suppress=True)

# load test data
test_data_file = open("mnist_test.csv",'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# score of the NN
scorecard = []

# go through the test data and calculate score of NN
for record in test_data_list:
    # split record by ','
    all_values = record.split(',')
    # correct answer is the first number
    correct_label = int(all_values[0])
    # scale and shift inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = nn.query(inputs)
    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    # append correct or incorrect to list
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)

# calculate the performance score
scorecard_array = numpy.asarray(scorecard)

print("Performance = {}; Learning Rate = {}; Epochs = {}".format((float(scorecard_array.sum()) / scorecard_array.size),learning_rate,epochs))
