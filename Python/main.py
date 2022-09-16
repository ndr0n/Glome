import numpy as np
import time
import os
from osc import OscServer
from keras1 import NeuralNetRegression

#parameters
inputSize = 1200
outputSize = 1200
nHidden = 3
nNodes = 10
epochs = 100

trained = 0
nExamples = 0
x = np.array([0])
y = np.array([0])
nn = NeuralNetRegression(np.zeros((1,inputSize)), np.zeros((1,outputSize)), nHidden, nNodes)
oscserver = OscServer("127.0.0.1", 6448, inputSize, outputSize)
print("press ctrl+c: quit")

while True:
    time.sleep(0.1)
    if trained == 1:
        pred = np.array([oscserver.pred])
        if(pred.shape[1] == outputSize):
            yout = nn.predict(pred)
            yout = yout.tolist()
            msg = ''
            for value in yout:
                msg = msg + ' ' + str(value)
            msg = msg.replace(",", "")
            msg = msg.replace("[", "")
            msg = msg.replace("]", "")
            os.system("echo '" + msg + ";' | pdsend 3000")
    if oscserver.addexample == 1:
        if(oscserver.xin.size == inputSize & oscserver.yin.size == outputSize):
            if(nExamples==0):
                x = oscserver.xin
                y = oscserver.yin
            else:
                x = np.vstack((x,oscserver.xin))
                y = np.vstack((y,oscserver.yin))
            nExamples += 1
            print("nExamples:", nExamples)
        else:
            print("Error Adding Example - Wrong Size! xSize: " + str(oscserver.xin.size) + " | ySize: " + str(oscserver.yin.size))
        oscserver.addexample = 0
        pass
    if oscserver.delexample == 1:
        nExamples -= 1
        x = x[:-1]
        y = y[:-1]
        print(x)
        print(y)
        if(nExamples<0):nExamples=0
        print("Removed Example")
        print("nExamples:", nExamples)
        oscserver.delexample = 0
        pass
    if oscserver.delall == 1:
        nExamples = 0
        x = np.array([0])
        y = np.array([0])
        print("Cleared Examples")
        print("nExamples:", nExamples)
        oscserver.delall = 0
    if oscserver.train == 1:
        # train
        if(nExamples > 1):
            print("Training Neural Network...")
            print("nExamples:",nExamples,"Epochs:",oscserver.epochs)
            nn.train(x,y,nExamples,oscserver.epochs)
            print("Finished Training Neural Network")
            trained = 1
            oscserver.delall = 1
            oscserver.train = 0
        else:
            print("Error Training - Not Enough Examples.")
            oscserver.train = 0
        pass
    if oscserver.trainNew == 1:
        # train New
        if(nExamples > 1):
            nn = NeuralNetRegression(x,y,nHidden,nNodes)
            print("Training New Neural Network...")
            print("nExamples:",nExamples,"Epochs:",oscserver.epochs)
            nn.train(x,y,nExamples,oscserver.epochs)
            print("Finished Training New Neural Network")
            trained = 1
            oscserver.trainNew = 0
        else:
            print("no Examples found. Need atleast 2 Examples to Train")
            oscserver.trainNew = 0
        pass
    if oscserver.quit == 1:
        oscserver.server.shutdown()
        quit()
        break
    else:
        pass