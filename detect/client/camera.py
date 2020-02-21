import pyrealsense2 as rs
import numpy as np
class frame:
  
 def to_arr(self): 
  self.frame_arr = np.asanyarray(self.data)
  
  return self.frame_arr
 def to_data(self): 
  self.data = self.frame.get_data()
  
  return self.data
  
class color_frame(frame):
 def __init__(self,color_frame):
  self.frame = color_frame
 def roi(self,min_y,max_y,min_x,max_x): 
  
  resized = self.frame_arr[int(min_y):int(max_y), int(min_x):int(max_x)]
   
  return resized

class depth_frame(frame):
 def __init__(self,depth_frame):
  self.frame = depth_frame
 
 def get_depth_by_coords(self,x,y): 
   depth =  self.frame_arr[int(y)+140][int(x)+150]
   
   return depth

class Camera:
 def __init__(self):
  self.pipeline = rs.pipeline()

  config = rs.config()
  config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
  config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

  # Start streaming
  self.pipeline.start(config)
   
 def init_frames(self): 
  self.frames = self.pipeline.wait_for_frames()
 
 def color_frame(self): 
  
  self.color_frame_ = color_frame(self.frames.get_color_frame())
  
 def depth_frame(self): 
  
  self.depth_frame_ = depth_frame(self.frames.get_depth_frame())
   
 def get_depth_by_coords(self,x,y): 
  
  return self.depth_frame_.get_depth_by_coords(int(y),int(x))

 def roi(self,min_y,max_y,min_x,max_x): 
  return self.color_frame_.roi(int(min_y),int(max_y),int(min_x),int(max_x))

 def color_frame_to_arr(self): 
  return self.color_frame_.to_arr()

 def depth_frame_to_arr(self): 
  return self.depth_frame_.to_arr()

 def color_frame_to_data(self): 
  return self.color_frame_.to_data()

 def depth_frame_to_data(self): 
  return self.depth_frame_.to_data()






