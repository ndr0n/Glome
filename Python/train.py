import numpy as np
from net import NeuralNetRegression
import liblo

size = 8192
nHidden = 3
nNodes = 10

class OscServer:
    def __init__(self, ip, port):
        self.epochs = 100
        self.learn = True
        self.training = False
        self.nExamples = 0
        self.x = np.array([0])
        self.y = np.array([0])
        self.xin = np.array([0])
        self.yin = np.array([0])
        self.yout = np.array([0])
        self.model = NeuralNetRegression(np.zeros((1, size)), np.zeros((1, size)), nHidden, nNodes)
        self.server = liblo.Server(port)
        self.server.add_method("/keras/xin", None, self.getX)
        self.server.add_method("/keras/delexample", None, self.delExample)
        self.server.add_method("/keras/delall", None, self.delAll)
        self.server.add_method("/keras/train", None, self.trainModel)
        self.server.add_method("/keras/trainnew", None, self.trainNewModel)
        self.server.add_method("/keras/learn", None, self.setLearn)
        self.server.add_method(None, None, self.Fallback)

    def Fallback(self, path, args, types, src):
        print("got unknown message '%s' from '%s'" % (path, src.url))
        for a, t in zip(args, types):
            print("argument of type '%s': %s" % (t, a))

    def getX(self, path, args):
        if self.nExamples == 0:
            self.xin = np.array(args)
        else:
            self.xin = self.yin
        self.yin = np.array(args)
        if (self.learn == True) & (self.training == False):
            if self.nExamples == 0:
                self.x = np.reshape(self.xin, (1, size))
                self.y = np.reshape(self.yin, (1, size))
            else:
                self.x = np.vstack((self.x, self.xin))
                self.y = np.vstack((self.y, self.yin))
            self.nExamples = self.x.shape[0]
            print("nExamples:", self.nExamples)

    def setLearn(self, path, args):
        if args[0] == 1.0: self.learn = True
        else: self.learn = False

    def delExample(self, path, args):
        self.nExamples -= 1
        self.x = self.x[:-1]
        self.y = self.y[:-1]
        print(self.x)
        print(self.y)
        if self.nExamples < 0:
            self.nExamples = 0
        print("Removed Example")
        print("nExamples:", self.nExamples)

    def delAll(self, path, args):
        self.nExamples = 0
        x = np.array([0])
        y = np.array([0])
        print("Cleared Examples")
        print("nExamples:", self.nExamples)

    def trainModel(self, path, args):
        self.trainNetwork(False)

    def trainNewModel(self, path, args):
        self.trainNetwork(True)

    def trainNetwork(self, new):
        if self.nExamples > 1:
            self.training = True
            if new == True:
                self.model = NeuralNetRegression(self.x, self.y, nHidden, nNodes)
                print("Training New Neural Network...")
            else:
                print("Training Neural Network...")
            print("nExamples:", self.nExamples, "Epochs:", self.epochs)
            self.y = self.y[0:self.x.shape[0]]
            self.x = self.x[0:self.y.shape[0]]
            mjson = self.model.fit(self.x, self.y, self.epochs)
            print("Finished Training Neural Network")
            self.SaveModel(mjson)
            self.nExamples = 0
            self.x = np.array([0])
            self.y = np.array([0])
            print("Cleared Examples")
            print("nExamples:", self.nExamples)
            self.training = False
            self.server.send(6449, "/keras/load", [0,0])
        else:
            print("no Examples found. Need atleast 2 Examples to Train")

    def SaveModel(self, mj):
        # serialize model to JSON
        model_json = mj.to_json()
        with open("glome.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        mj.save_weights("glome.h5")
        print("Saved model to disk")

oscserver = OscServer("127.0.0.1", 6448);
print("press ctrl+c: quit")

try:
    while True:
        oscserver.server.recv(1000);
        pass;
except KeyboardInterrupt:
    oscserver.server.stop();
    quit()