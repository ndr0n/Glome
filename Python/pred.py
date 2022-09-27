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
from tensorflow.keras.models import Sequential, model_from_json
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


class OscPredict:
    def __init__(self, ip, port):
        self.trained = False
        self.yout = np.array([0])
        self.dispatcher = dispatcher.Dispatcher()
        self.dispatcher.map('/keras/predict', self.Predict)
        self.dispatcher.map('/keras/load', self.Load)
        self.server = AsyncIOOSCUDPServer((ip, port), self.dispatcher, asyncio.get_event_loop())
        self.model = NeuralNetRegression(np.zeros((1, size)), np.zeros((1, size)), nHidden, nNodes)
        self.oscclient = OscClient("127.0.0.1", 3000)

    async def StartServer(self):
        self.transport, self.protocol = await self.server.create_serve_endpoint()

    def Predict(self, unused_addr, *args):
        if self.trained == True:
            self.yout = self.model.predict(np.reshape(np.array(args), (1, size)))
            self.yout = np.reshape(self.yout, (round(size/256), 256))
            for chunk in self.yout:
                self.oscclient.sendMsg(chunk.tolist(), '/keras/yout')
            self.oscclient.sendMsg([1], '/keras/sent')
            # self.oscclient.sendMsg(self.yout.tolist(), '/keras/yout')

    def Load(self, unused_addr, *args):
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

async def loop(predict):
    try:
        while True:
            await asyncio.sleep(1);
            pass
    except KeyboardInterrupt:
        print('Closing...')
        predict.transport.close()
        predict.server.shutdown()
        quit()

async def init_main():
    print("press ctrl+c: quit")
    oscpredict = OscPredict("127.0.0.1", 6449)
    await oscpredict.StartServer()
    await loop(oscpredict)

asyncio.run(init_main())