import numpy as np
import matplotlib.pyplot as plot
import random as rand
np.set_printoptions(suppress=True)

class NeuralNetwork(object):
    def __init__(self, bias=1, InputNodes=1, HiddenNeurons=10, OutputNeurons=1, eta=0.05, momentum=0.05):
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
                self.weightsHidden[j][i] = rand.uniform(-5, 5)

        # # losowe wagi dla warstwy outputu
        self.weightsOutput = np.zeros((self.hiddenNeurons + self.bias, self.outputNeurons))
        for i in range(self.outputNeurons):
            for j in range(self.hiddenNeurons + self.bias):
                self.weightsOutput[j][i] = rand.uniform(-5, 5)

        # zdefiniowanie wektorow przechowujacych outputy i zarazem inputy dla kolejenej warstwy
        self.aInput = np.zeros(self.inputNodes)
        self.aHidden = np.zeros(self.hiddenNeurons + self.bias)
        self.aOutput = np.zeros(self.outputNeurons)

        # zdefiniowanie wektorow delt wykorzystywanych przy wstecznej propagacji
        self.deltaHidden = np.zeros(self.hiddenNeurons)
        self.deltaOutput = np.zeros(self.outputNeurons)

        # zdefiniowanie wektorów poprzedniej wartości - wykorzystywane przy obliczniu momentum
        self.previousDeltaChangeHidden = np.array(self.weightsHidden)
        self.previousDeltaChangeOutput = np.array(self.weightsOutput)

        # zdefiniowanie tablic przzechowujacych punkty dla wykresów
        self.epochsPoints = []
        self.errorsPoints = []
        self.errorsTest = []
        self.pointX = []
        self.targetY = []
        self.testX = []
        self.testY = []
        self.x = []
        self.y = []

    def trainNeuralNetwork(self, data, inputsTest, epochs=1000):
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
                    self.pointX.append(input)
                    self.targetY.append(target)
            epochError /= data.shape[0]
            print('Error: ', epochError)
            # testError = self.checkQuality(inputsTest)
            # print("Error of aproximation: ", testError)
            # self.errorsTest.append(testError)
            self.epochsPoints.append(i)
            self.errorsPoints.append(epochError)
        # Wykres bledu sredniokwadratowego dla zbioru treningowego
        plot.plot(self.epochsPoints, self.errorsPoints, "pink")
        plot.title('Chart for '+self.hiddenNeurons.__str__()+' neurons - errors from training input')
        plot.xlabel('Epochs')
        plot.ylabel('Error')
        #plot.savefig(self.momentum.__str__() + 'err2.png')
        plot.show()

        #Wykres bledu sredniokwadratowego dla zbioru testowego
        # plot.plot(self.epochsPoints, self.errorsTest, "green")
        # plot.title('Chart for ' + self.hiddenNeurons.__str__() + ' neurons - errors from testing input')
        # plot.xlabel('Epochs')
        # plot.ylabel('Error')
        # #plot.savefig(self.hiddenNeurons.__str__() + 'errorsTesting2')
        # plot.show()

        #Wykres uzyskanej funkcji wraz z naniesionym punktami treningowymi
        plot.plot(self.pointX, self.targetY, 'ro', label='Training Points')
        for i in range(-4500, 4500):
            self.x.append(i/1000)
            self.y.append(float(self.query(i/1000)))
        plot.plot(self.x, self.y, label='Aproximated function:\n'+ 'eta= '+ self.eta.__str__() + ' \nmomentum= '
                                                              + self.momentum.__str__() + '\nbias= '+self.bias.__str__())
        plot.title('Chart for ' + self.hiddenNeurons.__str__()+' neurons')
        plot.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), shadow=True, ncol=2)
        plot.xticks(np.arange(-5, 6, 1))
        plot.ylim(-8, 4)
        #plot.savefig(self.momentum.__str__()+'fun2.png')
        plot.show()

        #Wykres uzyskanej funkcji wraz z naniesionym punktami testowymi
        plot.title('Chart for ' + self.hiddenNeurons.__str__() + ' neurons')
        for i in inputsTest:
            input = i[0:self.inputNodes-self.bias]
            target = i[self.inputNodes-self.bias:]
            self.testX.append(input)
            self.testY.append(target)
        plot.plot(self.testX, self.testY, "yo", label='Testing points')
        plot.plot(self.x, self.y, label='Aproximated function')
        plot.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), shadow=True, ncol=2)
        plot.xticks(np.arange(-5, 6, 1))
        plot.ylim(-8, 4)
        #plot.savefig(self.hiddenNeurons.__str__() + 'functionTestingPoints2')
        plot.show()

    def backPropagation(self, target, output):
        # wsteczna propagacja - zaczynamy od ostatniej warstwy
        # obliczenie delt dla warstwy outputu
        for i in range(self.outputNeurons):
            self.deltaOutput[i] = (output[i] - target[i]) * 1

        # zmiana wag dla warstwy outputu
        for j in range(self.hiddenNeurons + self.bias):
            for i in range(self.outputNeurons):
                self.weightsOutput[j][i] = self.weightsOutput[j][i] - (self.deltaOutput[i] * self.aHidden[j] * self.eta) \
                                           + self.momentum * (self.weightsOutput[j][i] - self.previousDeltaChangeOutput[j][i])
                self.previousDeltaChangeOutput[j][i] = self.weightsOutput[j][i]


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
                                           + self.momentum * (self.weightsHidden[i][j] - self.previousDeltaChangeHidden[i][j])
                self.previousDeltaChangeHidden[i][j] = self.weightsHidden[i][j]

        # Obliczenie bledu srednio kwadratowego
        error = self.calculateError(target, output)
        return error

    def feedForward(self, input):
        # przepisanie inputu do pomocniczego wektora

        for i in range(self.inputNodes - self.bias):
            self.aInput[i] = float(input[i])

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
            self.aOutput[j] = sum
        return self.aOutput

    def query(self, x):
        self.aInput[0] = x
        output = self.feedForward(self.aInput)
        return output

    def checkQuality(self, testInput):
        error = 0
        for i in testInput:
            input = i[0:self.inputNodes-self.bias]
            target = i[self.inputNodes-self.bias:]
            output = self.feedForward(input)
            error += self.calculateError(target, output)
        error /= testInput.shape[0]
        return error

    def calculateError(self, target, output):
        errorAvg = 0
        errors = np.zeros(self.outputNeurons)
        for i in range(self.outputNeurons):
            errors[i] = output[i] - target[i]
        for i in range(self.outputNeurons):
            errorAvg += pow(errors[i], 2)
        errorAvg /= 2
        return errorAvg

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoidDerivative(self, x):
        return x * (1 - x)

# odczytanie pliku approximation_train1
with open('approximation_train1.txt') as f:
    w, h = [float(x) for x in next(f).split()]
    inputsTrain1 = []
    for line in f:
        inputsTrain1.append([float(x) for x in line.split()])
inputsTrain1 = np.array(inputsTrain1)
# odczytanie pliku approximation_train2
with open('approximation_train2.txt') as f:
    w, h = [float(x) for x in next(f).split()]
    inputsTrain2 = []
    for line in f:
        inputsTrain2.append([float(x) for x in line.split()])
inputsTrain2 = np.array(inputsTrain2)
# odczytanie pliku approximation_test
with open('approximation_test.txt') as f:
    w, h = [float(x) for x in next(f).split()]
    inputsTest = []
    for line in f:
        inputsTest.append([float(x) for x in line.split()])
inputsTest = np.array(inputsTest)

network1 = NeuralNetwork()
network1.trainNeuralNetwork(inputsTrain1, inputsTest)

# network2 = NeuralNetwork()
# network2.trainNeuralNetwork(inputsTrain2, inputsTest)













