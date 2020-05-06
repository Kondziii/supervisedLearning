import numpy as np
import matplotlib.pyplot as plot
import random as rand
np.set_printoptions(suppress=True)


class NeuralNetwork(object):
    def __init__(self, bias=1, InputNodes=4, HiddenNeurons=3, OutputNeurons=4, eta=0.5, momentum=0.5):
        self.bias = bias
        self.inputNodes = InputNodes + self.bias
        self.hiddenNeurons = HiddenNeurons
        self.outputNeurons = OutputNeurons

        # określenie parametrów nauki
        self.eta = eta
        self.momentum = momentum

        # losowe wagi dla warstwy ukrytej
        self.weightsHidden = np.zeros((self.inputNodes, self.hiddenNeurons))
        for i in range(self.hiddenNeurons):
            for j in range(self.inputNodes):
                self.weightsHidden[j][i] = rand.uniform(-1, 1)

        # # losowe wagi dla warstwy outputu
        self.weightsOutput = np.zeros((self.hiddenNeurons + self.bias, self.outputNeurons))
        for i in range(self.outputNeurons):
            for j in range(self.hiddenNeurons + self.bias):
                self.weightsOutput[j][i] = rand.uniform(-1, 1)

        # zdefiniowanie wektorow przechowujacych outputy i zarazem inputy dla kolejenej warstwy
        self.aInput = np.zeros(self.inputNodes)
        self.aHidden = np.zeros(self.hiddenNeurons + self.bias)
        self.aOutput = np.zeros(self.outputNeurons)

        # zdefiniowanie wektorow delt wykorzystywanych przy wstecznej propagacji
        self.deltaHidden = np.zeros(self.hiddenNeurons)
        self.deltaOutput = np.zeros(self.outputNeurons)

        # zdefiniowanie wektorów poprzedniej wartości - wykorzystywane przy obliczniu momentum
        self.previousWeightHidden = np.array(self.weightsHidden)
        self.previousWeightOutput = np.array(self.weightsOutput)
        # zdefiniowanie tablic przzechowujacych punkty dla wykresu
        self.epochsPoints = []
        self.errorsPoints = []

    def trainNeuralNetwork(self, data, epochs=1000):
        for i in range(epochs):
            epochError = 0
            np.random.shuffle(data)  #stosujemy nauczanie online dlatego zalecane jest mieszanie danych po kazdej epoce
            for d in data:
                input = d[0:self.inputNodes-self.bias]
                target = d[self.inputNodes-self.bias:]
                output = self.feedForward(d)
                epochError += self.backPropagation(target, output)
                if i == epochs - 1:
                    print('For input: ', input, 'Outcome: ', output, 'Target: ', target)
            epochError /= data.shape[0]
            print('Error: ', epochError)
            self.epochsPoints.append(i)
            self.errorsPoints.append(epochError)
        self.showPlot()

    def backPropagation(self, target, output):
        errorAvg = 0
        errors = np.zeros(self.outputNeurons)
        # wsteczna propagacja - zaczynamy od ostatniej warstwy
        # obliczenie delt dla warstwy outputu
        for i in range(self.outputNeurons):
            errors[i] = output[i] - target[i]
            self.deltaOutput[i] = errors[i] * self.sigmoidDerivative(self.aOutput[i])

        # zmiana wag dla warstwy outputu
        for j in range(self.hiddenNeurons + self.bias):
            for i in range(self.outputNeurons):
                self.weightsOutput[j][i] = self.weightsOutput[j][i] - (
                            self.deltaOutput[i] * self.aHidden[j] * self.eta) + self.momentum * (
                                                       self.weightsOutput[j][i] - self.previousWeightOutput[j][i])
                self.previousWeightOutput[j][i] = self.weightsOutput[j][i]
        # teraz kolejna warstwa, czyli warstwa ukryta - obliczenie delt
        for i in range(self.hiddenNeurons):
            errorHidden = 0.0
            for j in range(self.outputNeurons):
                errorHidden += self.weightsOutput[i][j] * self.deltaOutput[j]
            self.deltaHidden[i] = errorHidden * self.sigmoidDerivative(self.aHidden[i])

        # zmiana wag dla warstwy ukrytej
        for i in range(self.inputNodes):
            for j in range(self.hiddenNeurons):
                self.weightsHidden[i][j] = self.weightsHidden[i][j] - (self.deltaHidden[j] * self.aInput[i] * self.eta)\
                                           + self.momentum * (self.weightsHidden[i][j] - self.previousWeightHidden[i][j])
                self.previousWeightHidden[i][j] = self.weightsHidden[i][j]

        # Obliczenie bledu srednio kwadratowego
        for i in range(self.outputNeurons):
            errorAvg += pow(errors[i], 2)
        errorAvg /= 2
        return errorAvg

    def feedForward(self, input):
        # przepisanie inputu do pomocniczego wektora
        for i in range(self.inputNodes - self.bias):
            self.aInput[i] = input[i]

        # Dodanie biasu do inputu na warstwe hidden
        if self.bias == 1:
            self.aInput[self.inputNodes - self.bias] = 1

        # Obliczenie aktywacji dla warstwy ukrytej
        for j in range(self.hiddenNeurons):
            sum = 0.0
            for i in range(self.inputNodes):
                sum += self.aInput[i] * self.weightsHidden[i][j]
            self.aHidden[j] = self.sigmoid(sum)

        # Dodanie biasu do inputu na warstwe output
        if self.bias == 1:
            self.aHidden[self.hiddenNeurons] = 1

        # Obliczenie aktywacji dla warstwy outputu
        for j in range(self.outputNeurons):
            sum = 0.0
            for i in range(self.hiddenNeurons + self.bias):
                sum += self.aHidden[i] * self.weightsOutput[i][j]
            self.aOutput[j] = self.sigmoid(sum)
        return self.aOutput

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoidDerivative(self, x):
        return x * (1 - x)

    def showPlot(self):
        plot.plot(self.epochsPoints, self.errorsPoints, "blue")
        plot.xlabel('Epochs')
        plot.ylabel('Error')
        plot.plot(self.epochsPoints, self.errorsPoints, label='eta= '+ self.eta.__str__() + ' \nmomentum= '
                                                              + self.momentum.__str__() + '\nbias= '+self.bias.__str__())
        plot.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), shadow=True, ncol=2)
        plot.title('Chart for ' + self.hiddenNeurons.__str__() + ' neurons')
        plot.ylim(0, 0.8)
        plot.show()


# wczytanie pliku transformation.txt, czyli określenie wejścia (zbioru uczącego)
inputs = np.zeros((4, 4))
file = open("transformation.txt", "r")
j = 0
for line in file.readlines():
    numbers = line.strip().split()
    inputs[j] = numbers
    j += 1

# zdefiniowanie wartosci oczekiwanych na wyjsciu - oczekujemy takich samych jak na wejsciu
expected = inputs
inputs = np.concatenate((inputs, expected), axis=1)

network = NeuralNetwork()
network.trainNeuralNetwork(inputs)


