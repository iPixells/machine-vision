from server_socket import server_Socket
from custom_mrcnn import custom_Mrcnn
from custom_config import custom_Config

if __name__ == '__main__':
     
     
     custom_config = custom_Config()
     custom_config.display()
      
     server_socket = server_Socket()
     
     custom_mrcnn = custom_Mrcnn(custom_config,server_socket)

     while True:
      print("run")
      
      img = server_socket.run()
      custom_mrcnn.run(img)
      
     server_socket.close()
     print("server finish")
