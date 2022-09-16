import argparse
import threading
import numpy as np
from pythonosc import dispatcher
from pythonosc import udp_client
from pythonosc import osc_server
from pythonosc.osc_message_builder import OscMessageBuilder


class OscClient:
    inputSize = 1200
    outputSize = 1200

    def __init__(self,ip,port, input, output):
        self.ip = ip
        self.port = port
        self.inputSize = input
        self.outputSize = output
        # self.address = address
        parser = argparse.ArgumentParser()
        parser.add_argument("--ip", default=ip)
        parser.add_argument("--port", type=int, default=port)
        args = parser.parse_args()
        self.client = udp_client.SimpleUDPClient(args.ip, args.port)

    def sendMsg(self,msg,msgAdress):
        builder = OscMessageBuilder(address=msgAdress)
        for v in msg[0]:
            builder.add_arg(v)
        out = builder.build()
        # print("sent osc message to",self.ip,"on port",self.port,"with address",self.address)
        self.client.send(out)

class OscServer:
    inputSize = 2400
    outputSize = 2400

    def getPred(self,unused_addr,*args):
        self.pred = np.array(args)
        # if arraynum == 0:
        #     self.prePred = args
        # if arraynum >= 1:
        #     self.prePred = np.vstack((self.prePred,args))
        # if arraynum == 14:
        #     test = self.prePred.ravel()
        #     if(test.size == self.outputSize):
        #         self.pred = test
        #         print("P:",self.pred.size)
        #     else:
        #         print("Error Receiving P - Wrong Size")

    def getX(self,unused_addr, *args):
        self.xin = np.array(args)
        # if arraynum == 0:
        #     self.prex = args
        # if arraynum >= 1:
        #     self.prex = np.vstack((self.prex,args))
        # if arraynum == 14:
        #     test = self.prex.ravel()
        #     if(test.size == self.inputSize):
        #         self.xin = test
        #         print("X:", self.xin.size)
        #     else:
        #         print("Error Receiving X - Wrong Size")

    def getY(self,unused_addr, *args):
        self.yin = np.array(args)
        # if arraynum == 0:
        #     self.prey = args
        # if arraynum >= 1:
        #     self.prey = np.vstack((self.prey,args))
        # if arraynum == 14:
        #     test = self.prey.ravel()
        #     if(test.size == self.outputSize):
        #         self.yin = test
        #         print("Y:", self.yin.size)
        #     else:
        #         print("Error Receiving Y - Wrong Size")
    
    def addExample(self,unused_addr,*args):
        self.addexample = 1
    
    def delExample(self,unused_addr,*args):
        self.delexample = 1
    
    def delAll(self,unused_addr,*args):
        self.delall = 1
    
    def trainModel(self,unused_addr,*args):
        self.train = 1

    def trainNewModel(self,unused_addr,*args):
        self.trainNew = 1

    def close(self,unused_addr,*args):
        self.quit = 1

    def getEpochs(self,*args):
        self.epochs = args[1]
        print("Epochs:",self.epochs)
        
    def __init__(self, ip, port, input, output):
        self.inputSize = input
        self.outputSize = output
        self.epochs = 100
        self.addexample = 0
        self.delexample = 0
        self.delall = 0
        self.train = 0
        self.trainNew = 0
        self.quit = 0
        self.pred = np.array([0])
        self.xin = np.array([0])
        self.yin = np.array([0])
        self.prePred = np.array([0])
        self.prex = np.array([0])
        self.prey = np.array([0])
        self.dispatcher = dispatcher.Dispatcher()
        self.dispatcher.map('/keras/xin', self.getX)
        self.dispatcher.map('/keras/yin',self.getY)
        self.dispatcher.map('/keras/predin',self.getPred)
        self.dispatcher.map('/keras/epochs',self.getEpochs)
        self.dispatcher.map('/keras/addexample',self.addExample)
        self.dispatcher.map('/keras/delexample',self.delExample)
        self.dispatcher.map('/keras/delall',self.delAll)
        self.dispatcher.map('/keras/train',self.trainModel)
        self.dispatcher.map('/keras/trainnew',self.trainNewModel)
        self.dispatcher.map('/keras/quit',self.close)
        self.server = osc_server.ThreadingOSCUDPServer((ip, port), self.dispatcher)
        server_thread = threading.Thread(target=self.server.serve_forever)
        server_thread.start()