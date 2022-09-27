import numpy as np
from net import NeuralNetRegression
from tensorflow.keras.models import Sequential, model_from_json
import liblo

size = 8192
nHidden = 3
nNodes = 10

class OscPredict:
    def __init__(self, ip, port):
        self.trained = False
        self.yout = np.array([0])
        self.model = NeuralNetRegression(np.zeros((1, size)), np.zeros((1, size)), nHidden, nNodes)
        self.server = liblo.Server(port)
        self.server.add_method("/keras/predict", None, self.Predict)
        self.server.add_method("/keras/load", None, self.Load)
        self.server.add_method(None, None, self.Fallback)

    def Fallback(self, path, args, types, src):
        print("got unknown message '%s' from '%s'" % (path, src.url))
        for a, t in zip(args, types):
            print("argument of type '%s': %s" % (t, a))

    def Predict(self, path, args):
        if self.trained == True:
            self.yout = self.model.predict(np.reshape(np.array(args), (1, size)))
            self.yout = np.reshape(self.yout, (round(size/256), 256))
            for chunk in self.yout:
                self.server.send(3000, '/keras/yout', chunk.tolist())
            self.server.send(3000, '/keras/sent', [1])

    def Load(self, path, args):
        self.LoadModel()

    def LoadModel(self):
        # load json and create model
        json_file = open('glome.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights("glome.h5")
        print("Loaded model from disk")
        self.trained = True


oscpredict = OscPredict("127.0.0.1", 6449);
print("press ctrl+c: quit")

try:
    while True:
        oscpredict.server.recv(1000);
        pass;
except KeyboardInterrupt:
    oscpredict.server.stop();
    quit()