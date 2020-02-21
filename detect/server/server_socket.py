import socket
import pickle
import numpy as np
import struct ## new
import zlib
import cv2


class server_Socket():
     
    def __init__(self):
     HOST=''
     PORT=8485
     self.data = b""

     self.s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)

     self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
     print('Socket created')

     self.s.bind((HOST,PORT))
     print('Socket bind complete')

     self.s.listen(0)
     print('Socket now listening')

     self.conn,addr=self.s.accept()
     
     self.payload_size = struct.calcsize(">L")

     print("payload_size: {}".format(self.payload_size))
 
     self.img_counter = 0
     
    def run(self):
      
      while len(self.data) < self.payload_size:
        print("Recv: {}".format(len(self.data)))
        self.data += self.conn.recv(4096)

      print("Done Recv: {}".format(len(self.data)))
      packed_msg_size = self.data[:self.payload_size]
      self.data = self.data[self.payload_size:]
      msg_size = struct.unpack(">L", packed_msg_size)[0]
      print("msg_size: {}".format(msg_size))

      while len(self.data) < msg_size:
        self.data += self.conn.recv(4096)

      self.frame_data = self.data[:msg_size]
      self.data = self.data[msg_size:]

      frame=pickle.loads(self.frame_data, fix_imports=True, encoding="bytes")
      #frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
      
      return frame 
    def send(self,data,attrs):
       
      dic = np.array(attrs[0:15])
      
      self.conn.sendall(pickle.dumps(dic, 0))
      #thisdict = pickle.dumps(thisdict, 0)
      #size = len(data)

      #print("{}: {}".format(self.img_counter,size))

      data = pickle.dumps(data, 0)
      size = len(data)
      print("size {}".format(size))
      self.conn.sendall(struct.pack(">L", size)+data)        
      self.img_counter += 1          
      
      #cv2.imshow('ImageWindow',frame)
      #cv2.waitKey(10)
    def close(self):
     self.s.close()
      
      
     
      
      
      
      
