from window import *
from client_socket import *
from camera import *

if __name__ == '__main__':
     client_socket = client_Socket()
     
     camera = Camera()
    
     app = Window(camera=camera,client_socket=client_socket)
     
     app.run()
     
          
 
