import argparse
import threading
from pythonosc import dispatcher
from pythonosc import udp_client
from pythonosc import osc_server
from pythonosc.osc_message_builder import OscMessageBuilder
import numpy as np
from net import NeuralNetRegression
from pythonosc.osc_server import AsyncIOOSCUDPServer
import asyncio
import time

size = 8192
nHidden = 3
nNodes = 10

class OscClient:
    def __init__(self,ip,port):
        self.ip = ip
        self.port = port
        parser = argparse.ArgumentParser()
        parser.add_argument("--ip", default=ip)
        parser.add_argument("--port", type=int, default=port)
        args = parser.parse_args()
        self.client = udp_client.SimpleUDPClient(args.ip, args.port)

    def sendMsg(self,msg,msgAdress):
        builder = OscMessageBuilder(address=msgAdress)
        for v in msg:
            builder.add_arg(v)
        out = builder.build()
        self.client.send(out)

class OscServer:
    def __init__(self, ip, port):
        self.epochs = 100
        self.learn = True
        self.training = False
        self.Examples = 0
        self.x = np.array([0])
        self.y = np.array([0])
        self.xin = np.array([0])
        self.yin = np.array([0])
        self.yout = np.array([0])
        self.dispatcher = dispatcher.Dispatcher()
        self.dispatcher.map('/keras/xin', self.getX)
        self.dispatcher.map('/keras/delexample', self.delExample)
        self.dispatcher.map('/keras/delall', self.delAll)
        self.dispatcher.map('/keras/train', self.trainModel)
        self.dispatcher.map('/keras/trainnew', self.trainNewModel)
        self.dispatcher.map('/keras/learn', self.setLearn)
        self.dispatcher.map('/keras/epochs', self.setEpochs)
        self.server = AsyncIOOSCUDPServer((ip, port), self.dispatcher, asyncio.get_event_loop())
        self.model = NeuralNetRegression(np.zeros((1, size)), np.zeros((1, size)), nHidden, nNodes)
        self.oscclient = OscClient("127.0.0.1", 6449)

    async def StartServer(self):
        self.transport, self.protocol = await self.server.create_serve_endpoint()

    def getX(self, unused_addr, *args):
        try:
            if self.Examples == 0:
                self.xin = np.array(args)
            else:
                self.xin = self.yin
            self.yin = np.array(args)
            if (self.learn == True) & (self.training == False):
                if self.Examples == 0:
                    self.x = np.reshape(self.xin, (1, size))
                    self.y = np.reshape(self.yin, (1, size))
                else:
                    self.x = np.vstack((self.x, self.xin))
                    self.y = np.vstack((self.y, self.yin))
                self.Examples = self.x.shape[0]
                print("Examples:", self.Examples)
        except Exception as e:
            print("Error while receiving example!");
            print(e);
            self.delAll("", [""]);

    def setLearn(self, unused_addr, *args):
        if args[0] == 1.0: self.learn = True
        else: self.learn = False

    def delExample(self, unused_addr, *args):
        self.Examples -= 1
        self.x = self.x[:-1]
        self.y = self.y[:-1]
        print(self.x)
        print(self.y)
        if self.Examples < 0:
            self.Examples = 0
        print("Removed Example")
        print("Examples:", self.Examples)

    def delAll(self, unused_addr, *args):
        self.Examples = 0
        x = np.array([0])
        y = np.array([0])
        print("Cleared Examples")
        print("Examples:", self.Examples)

    def trainModel(self, unused_addr, *args):
        self.trainNetwork(False)

    def trainNewModel(self, unused_addr, *args):
        self.trainNetwork(True)

    def setEpochs(self, unused_addr, *args):
        self.epochs = round(args[0])
        print("Set Epochs to", self.epochs);

    def trainNetwork(self, new):
        try:
            if self.Examples > 1:
                self.training = True
                if new == True:
                    self.model = NeuralNetRegression(self.x, self.y, nHidden, nNodes)
                    print("Training New Neural Network...")
                else:
                    print("Training Neural Network...")
                print("Examples:", self.Examples, "Epochs:", self.epochs)
                self.y = self.y[0:self.x.shape[0]]
                self.x = self.x[0:self.y.shape[0]]
                mjson = self.model.fit(self.x, self.y, self.epochs)
                print("Finished Training Neural Network!")
                self.SaveModel(mjson)
                self.Examples = 0
                self.x = np.array([0])
                self.y = np.array([0])
                print("Cleared Examples")
                print("Examples:", self.Examples)
                self.training = False
                self.oscclient.sendMsg([0,0], "/keras/load")
                time.sleep(0.1)
            else:
                print("Not enough examples. Need at least 2 examples to start training.")
        except Exception as e:
            print("Error while training network!");
            print(e);
            self.delAll("", [""]);
            self.training = False;


    def SaveModel(self, mj):
        # serialize model to JSON
        model_json = mj.to_json()
        with open("glome.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        mj.save_weights("glome.h5")
        print("Saved Neural Network!")

async def loop(server):
    try:
        while True:
            await asyncio.sleep(1);
            pass
    except KeyboardInterrupt:
        print('Closing...')
        server.transport.close()
        server.server.shutdown()
        quit()

async def init_main():
    print("press ctrl+c: quit")
    oscserver = OscServer("127.0.0.1", 6448)
    await oscserver.StartServer()
    await loop(oscserver)

asyncio.run(init_main())