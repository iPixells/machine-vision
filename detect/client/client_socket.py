import io
import socket
import struct
import time
import pickle
#import zlib
import numpy as np

class client_Socket():
  def __init__(self):
   self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
   self.client_socket.connect(('localhost', 8485))
   #connection = client_socket.makefile('wb')
   self.img_counter = 0
   self.payload_size = struct.calcsize(">L")
   
   #encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
  def run(self,data):
    
    #result, frame = cv2.imencode('.jpg', frame, encode_param)
    #data = zlib.compress(pickle.dumps(frame, 0))
    data = pickle.dumps(data, 0)
    size = len(data)

    print("{}: {}".format(self.img_counter, size))
    self.client_socket.sendall(struct.pack(">L", size) + data)
    self.img_counter += 1
  def rev(self):
   self.data = b""
   self.attr = b""
   
   while len(self.attr) < self.payload_size:
    print("Recv: {}".format(len(self.attr)))
    self.attr += self.client_socket.recv(4096)
   self.attr=pickle.loads(self.attr)
   print(self.attr)

   while len(self.data) < self.payload_size:
    print("Recv: {}".format(len(self.data)))
    self.data += self.client_socket.recv(4096)

   print("Done Recv: {}".format(len(self.data)))
   packed_msg_size = self.data[:self.payload_size]
   self.data = self.data[self.payload_size:]
   msg_size = struct.unpack(">L", packed_msg_size)[0]
   print("msg_size: {}".format(msg_size))

   while len(self.data) < msg_size:
    self.data += self.client_socket.recv(4096)

   frame_data = self.data[:msg_size]
   self.data = self.data[msg_size:]

   frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
      
   return self.attr,frame
  def close(self):
     self.client_socket.close()


