import tkinter as tk

from PIL import Image, ImageTk

import numpy as np
import cv2

xmin, xmax, ymin, ymax = 250, 420, 140, 290

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

init_max_x = 640

init_max_y = 480

def donothing():
   filewin = Toplevel(root)
   button = Button(filewin, text="Do nothing button")
   button.pack()
   
class Window():
    def __init__(self,camera,client_socket):
     global xmin, xmax, ymin, ymax
     self.client_socket = client_socket
     self.camera = camera

     self.main = tk.Tk()

     w, h = self.main.winfo_screenwidth(), self.main.winfo_screenheight()
     self.main.geometry("%dx%d+0+0" % (w, h))
     
     self.toolbar = tk.Frame(self.main, borderwidth=2, bg='slategray4', relief='raised')
     self.toolbar.pack(side=tk.TOP, fill=tk.X)

     self.button = tk.Button(self.toolbar, command=self.tb_click)      
     self.button.pack(side="left")
     self.button.configure(text="Start", background="Grey",padx=50)
     
     self.frame = tk.Frame(self.main)  
     self.frame.pack(side = tk.LEFT, fill=tk.Y)   

     self.roibar = tk.Frame(self.frame, borderwidth=2, bg='slategray4', relief='raised')
     self.roibar.pack(side=tk.TOP, fill=tk.X)
     
     self.label = tk.Label(self.roibar, text="min x")
     self.label.pack( side = tk.LEFT)

     validate_minx = (self.main.register(self.callback_minx),'%S')
     self.min_x = tk.Entry(self.roibar, bd =0, width=5,justify='right',validate="key", validatecommand=validate_minx)
     self.min_x.insert(tk.END,xmin)
     
     self.min_x.pack(side = tk.LEFT)

     self.label = tk.Label(self.roibar, text="max x")
     self.label.pack( side = tk.LEFT)
     validate_maxx = (self.main.register(self.callback_maxx),'%S')
     self.max_x = tk.Entry(self.roibar, bd =0, width=5,justify='right',validate="key", validatecommand=validate_maxx)
     self.max_x.insert(tk.END,xmax)
     self.max_x.pack(side = tk.LEFT)

     self.label = tk.Label(self.roibar, text="min y")
     self.label.pack( side = tk.LEFT)
     validate_miny = (self.main.register(self.callback_miny),'%S')
     self.min_y = tk.Entry(self.roibar, bd =0, width=5,justify='right',validate="key", validatecommand=validate_miny)
     self.min_y.insert(tk.END,ymin)
     self.min_y.pack(side = tk.LEFT)

     self.label = tk.Label(self.roibar, text="max y")
     self.label.pack( side = tk.LEFT)
     validate_maxy = (self.main.register(self.callback_maxy),'%S')
     self.max_y = tk.Entry(self.roibar, bd =0, width=5,justify='right',validate="key", validatecommand=validate_maxy)
     self.max_y.insert(tk.END,ymax)
     self.max_y.pack(side = tk.LEFT)
     
     self.canvas = tk.Canvas(self.frame,width=660,height=500)
     self.canvas.pack(side="top")
     
     self.img =  ImageTk.PhotoImage(file="img_1.jpg")

     self.imgArea = self.canvas.create_image(30,30, anchor="nw", image=self.img)

     self.listbox = tk.Listbox(self.frame,width="80")
     self.listbox.pack(side="top",padx=(40, 10),pady=(40, 10))
     """
     self.menubar = tk.Menu(self.main)

     filemenu = tk.Menu(self.menubar, tearoff = 0)
     filemenu.add_command(label="New", command = donothing)
     filemenu.add_command(label = "Open", command = donothing)
     filemenu.add_command(label = "Save", command = donothing)
     filemenu.add_command(label = "Save as...", command = donothing)
     filemenu.add_command(label = "Close", command = donothing)

     filemenu.add_separator()

     filemenu.add_command(label = "Exit", command = self.main.quit)
     self.menubar.add_cascade(label = "File", menu = filemenu)

     editmenu = tk.Menu(self.menubar, tearoff=0)
     editmenu.add_command(label = "Undo", command = donothing)

     editmenu.add_separator()

     editmenu.add_command(label = "Cut", command = donothing)
     editmenu.add_command(label = "Copy", command = donothing)
     editmenu.add_command(label = "Paste", command = donothing)
     editmenu.add_command(label = "Delete", command = donothing)
     editmenu.add_command(label = "Select All", command = donothing)

     self.menubar.add_cascade(label = "Edit", menu = editmenu)
     helpmenu = tk.Menu(self.menubar, tearoff=0)
     helpmenu.add_command(label = "Help Index", command = donothing)
     helpmenu.add_command(label = "About...", command = donothing)
     self.menubar.add_cascade(label = "Help", menu = helpmenu)

     self.main.config(menu = self.menubar)
     """
    def callback_minx(self,S):
     global xmin,number
     
     number_ = number
     
     if type(xmin) == int:
      number_.append(str(xmin))
       
     if S in number_:
        return True

     self.main.bell()  

     return False

    def callback_miny(self,S):
     global ymin,number 
     
     number_ = number
     
     if type(ymin) == int:
      number_.append(str(ymin))
    
     if S in number_:
        return True

     self.main.bell()  

     return False

    def callback_maxx(self,S):
     global xmax,init_max_x,number
     
     number_ = number

     if type(xmax) == int:
      number_.append(str(xmax))

     if S in number_:
        return True
     self.main.bell()  
     return False

    def callback_maxy(self,S):
     global ymax,init_max_y,number
     
     number_ = number

     if type(ymax) == int:
      number_.append(str(ymax))

     if S in number_:
        return True

     self.main.bell()  

     return False
      
    def tb_click(self):  
     global xmin, xmax, ymin, ymax
     width =  640  
     height = 480  
     dim = (width, height) 
     self.camera.init_frames()
     self.camera.color_frame()
     self.camera.color_frame_to_data()      
     self.camera.color_frame_to_arr() 
  
     self.camera.depth_frame()
     self.camera.depth_frame_to_data()      
     self.camera.depth_frame_to_arr() 
      
     # BB
     #xmin, xmax, ymin, ymax = 0, 640, 0, 480# BB
     
     min_x = self.min_x.get()
     min_y = self.min_y.get()
     max_x = self.max_x.get()
     max_y = self.max_y.get()
     
     resized = self.camera.roi(int(min_y),int(max_y), int(min_x),int(max_x))
     
     self.client_socket.run(resized)
     
     attrs,image = self.client_socket.rev()                         
     
     #resized = cv2.resize(image,dim, interpolation = cv2.INTER_AREA)
     self.img =  ImageTk.PhotoImage(image=Image.fromarray(image))
     
     self.canvas.itemconfig(self.imgArea, image =self.img)
      
     self.listbox.delete(0, tk.END)
     bg = 'gray90'
     
     n = 1
     for attr in attrs:
      depth = self.camera.get_depth_by_coords(attr['point'][0],attr['point'][1])
      self.listbox.insert("end","{}. center x ={:>7.1f} center y ={:>7.1f} angle = {:>} depth = {:>}".format(n,attr['point'][0],attr['point'][1],int(attr['angle']),depth))
       
      if bg == 'gray90':
       bg = ''
      elif bg == '':
       bg = 'gray90'
      self.listbox.itemconfig("end" ,{'bg':bg})
      n+=1
       
     
    def run(self):
          self.main.mainloop()   
           

            
