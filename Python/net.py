from keras.models import Sequential
from keras.layers import Dense

class NeuralNetRegression:
    def __init__(self,x,y,nHidden,nNodes):
        self.nHidden = nHidden
        self.nNodes = nNodes
        self.model = Sequential()
        self.model.add(Dense(self.nNodes, input_dim=x.shape[1], activation='linear'))
        for i in range(nHidden-1):
            self.model.add(Dense(self.nNodes, activation='relu'))
        self.model.add(Dense(y.shape[1], activation='linear'))

    def fit(self, x, y, epochs):
        self.model.compile(loss='mse', optimizer='adam')
        self.model.fit(x, y, epochs=epochs,batch_size=x.shape[0],verbose=0)

    def predict(self, xin):
        return self.model.predict(xin)