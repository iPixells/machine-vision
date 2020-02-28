"""
Mask R-CNN
Display and Visualization Functions.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import sys
import logging
import random
import itertools
import colorsys

import math

import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import IPython.display

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import cv2

import math

global_contours = None

pi = 3.1415926535897932384626433832795

divisor = 180.0

radian = pi/divisor


#ความกว้างของวัตถุ
obj_width = 22

#ความยาวของวัตถุ
obj_height = 37

#ความกว้างครึ่งหนึ่งของวัตถุ
half_width = obj_width//2

#ความยาวครึ่งหนึ่งของวัตถุ
half_height = obj_height//2


############################################################d
#  Visualization
############################################################

def display_images(images, titles=None, cols=4, cmap=None, norm=None,
                   interpolation=None):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    titles: optional. A list of titles to display with each image.
    cols: number of images per row
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interporlation to use for display.
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    plt.figure(figsize=(14, 14 * rows // cols))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=9)
        plt.axis('off')
        plt.imshow(image.astype(np.uint8), cmap=cmap,
                   norm=norm, interpolation=interpolation)
        i += 1
    plt.show()


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] ,
                                  image[:, :, c])
    
    #print("x")
    #print(image)
    return image


#function สำหรับ คำนวณหาองศา จาก จุด x,y
def get_theta(x,y):
    #print(x)
    #print(y)
    theta = math.atan(y*1.0/x) / (2*math.pi) * 360
    if x < 0:
        theta += 180
    if theta < 0:
        theta += 360
    
    #theta
    return 360-theta 

#function สำหรับ คำนวณหาองศา จาก จุด ความยาว และ องศา
def get_coordbyangle(length,angle):
    global radian

    angle_x =  length*math.cos(angle * radian);
    angle_y =  length*math.sin(angle * radian);
    
    return angle_x,angle_y

#function สำหรับ ลบจุด สอง จุด
def get_coord_target(x1,y1,x2,y2):
    x = x1 - x2
    y = y1 - y2
    return x,y
#function สำหรับ วาดเส้น สะท้อน องศา
def draw(x,y,angle,side,ax):
    
    if side == 'height':
     length = half_height
     result_angle = 180-angle
    elif side == 'width':
     length = half_width
     result_angle = 90-angle
    
    angle_x,angle_y =  get_coordbyangle(length,result_angle)
 
    x1,y1 = get_coord_target(x,y,angle_x,angle_y)

    ax.add_line(lines.Line2D([x,x1],[y,y1]))  

def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(1, 1), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None,
                      making_video=False, making_image=False, detect=False, hc=False, real_time=False):

    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
 
    global_contours = []
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = True
    if not ax:
        fig, ax = plt.subplots(1)#,figsize=figsize)
        
        fig.subplots_adjust(left=-1.2, right=2.2, top=1.1, bottom=-0.1, wspace=0, hspace=0)
        canvas = FigureCanvas(fig)
    
    # Generate random colors
    if not making_video or not real_time:
        colors = colors or random_colors(N)
    # Show area outside image boundaries.
    height, width = image.shape[:2]
    #ax.set_ylim(height + 10,0)
    #ax.set_xlim(0, width + 10)
    
    ax.axis('off')
    ax.set_title(title)
    
    masked_image = image.astype(np.uint32).copy()
    
    attrs = []
    new_contours = []
    ii = 0
    for i in range(N):
     # Mask
        mask = masks[:, :, i]
        
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        
        contours = find_contours(padded_mask, 0.5)
        if len(contours) > 0:
         verts = contours[0]
         ok = 1
         dummy_contours = []
         
        # Subtract the padding and flip (y, x) to (x, y)
         verts = np.fliplr(verts) - 1
            
         n = len(verts)
         
         if n < 20:
          continue
           
         print(len(new_contours))
         if len(new_contours) > 0:
          for cur_verts in new_contours:
           if ok == 1 and n  > cur_verts[1]:
            dummy_contours.append([i,n])
            ok = 0
          
           dummy_contours.append(cur_verts)
              
          if ok == 1:
           dummy_contours.append([i,n])

          #print(len(dummy_contours))
          new_contours = dummy_contours
         else:
          new_contours.append([i,n])
         #print("new_contours")
         #print(new_contours)
           
            #ax.add_line(lines.Line2D([1100,1100],[600,1200])) 
    #return
    for curr in new_contours:
        i = curr[0]
        
        print("i")
        print(curr)
        result_angle = 0
        class_id = class_ids[i]
        if making_video or real_time:
            # you can also assign a specific color for each class. etc:
            # if class_id == 1:
            #     color = colors[0]
            # elif class_id == 2:
            #     color = colors[1]
            color = colors[class_id-1]
        elif hc:
            #just for hard-code the mask for paper
            if class_id == 14:
                color = colors[0]
            else:
                color = colors[class_id]
        else:
            color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        
        #start x1 = 10,y1=300

        #ax.add_line(lines.Line2D([10,x2], [300, y2]))    
        #if show_bbox:
        """
        p = patches.Rectangle((0,0), x1, y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
        
        p1 = patches.Rectangle((0,0), x2, y2, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
         
        print(x1)
        print(y1)
        
        print(x2 - x1)
        print(y2 - y1)
        ax.add_patch(p)
        ax.add_patch(p1)
        """
        # Label
        
        if not captions:
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            x = random.randint(x1, (x1 + x2) // 2)
            caption = "{} {:.1f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ii+=1
        ax.text(x1, y1 + 8,caption+"no."+str(ii),color='w', size=10, backgroundcolor="none")
                
        # Mask
        mask = masks[:, :, i]
        
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)
        
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        
        contours = find_contours(padded_mask, 0.5)
        
        #global_contours.append(contours)
          
        #print(contours)
        #print("contours")
        #print(len(contours))
        
        for verts in contours:
            
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            
            n = len(verts)
         
            if n < 20:
             continue
           
            #ax.add_line(lines.Line2D([1100,1100],[600,1200])) 
            #ax.add_line(lines.Line2D([1628,1628],[0,927])) 
            ax.add_line(lines.Line2D([5,5],[15,30])) 
            ax.add_line(lines.Line2D([5,20],[15,25]))
            ax.add_line(lines.Line2D([5,25],[15,15]))
            attr = {}
            point = []
            print(n)
            
            n1 = n//2

            n2 = n1//2
            n6 = n - n//4
            
            n3 = n2//2

            n7 = n2 - n3//2
  
            n4 = n6//2
            n5 = n-n2//2

            n8 = n-n6//2

            n9 = n -n8//2
            """
            p1_1 = verts[0][0]
            p1_2 = verts[0][1]
             
            p2_1 = verts[n3][0]
            p2_2 = verts[n3][1]
            
            p3_1 = verts[n2][0]
            p3_2 = verts[n2][1]
            
            p4_1 = verts[n4][0]
            p4_2 = verts[n4][1]
            
            p5_1 = verts[n1][0]
            p5_2 = verts[n1][1]
            
            p6_1 = verts[n8][0]
            n6_2 = verts[n8][1]
            
            p7_1 = verts[n6][0]
            p7_2 = verts[n6][1]
            
            p8_1 = verts[n5][0]
            p8_2 = verts[n5][1]
            
            point = [p1_1,p1_2,p2_1,p2_2,p3_1,p3_2,p4_1,p4_2,p5_1,p5_2,p6_1,n6_2,p7_1,p7_2,p8_1,p8_2]
            """
            """
            print("1")
            print(verts[0][0])
            print(verts[0][1])
            print("2")   
            print(verts[n3][0])
            print(verts[n3][1])
            print("3")
            print(verts[n2][0])
            print(verts[n2][1])
            print("4")
            print(verts[n4][0])
            print(verts[n4][1])
            print("5")
            print(verts[n1][0])
            print(verts[n1][1])
            print("6")
            print(verts[n8][0])
            print(verts[n8][1])
            print("7")
            print(verts[n6][0])
            print(verts[n6][1])
            print("8")
            print(verts[n5][0])
            print(verts[n5][1])
            """
            print(verts[0])
            print(verts[1])
            """
            p1 = patches.Rectangle((0,0), verts[0][0], verts[0][1], linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
         
            """
            #print(verts[0][1] -  verts[n1][1])
            
            
            #ax.add_patch(p1)
            #ax.add_line(lines.Line2D([verts[0][0],verts[n1][0]],[verts[0][1],verts[n1][1]]))
            print("360")
            print(verts[0][1]-verts[n1][1])
            #ax.add_line(lines.Line2D([verts[0][0],verts[0][0]-250],[verts[0][1],verts[0][1]]))
            #ax.add_line(lines.Line2D([verts[0][0],verts[0][0]],[verts[0][1],verts[0][1]-380]))
            
            print("จุดต่ำสุด")
            print("x "+str(verts[0][0]))
            print("y "+str(verts[0][1]))
            
            """
            x1 = 0
            x2 = 0
            i = 1
            for vert in verts:
             if i == 10:
              i = 1
              x1 = vert[0]
              
              if x2 == 0 or x2 > x1:
               x2 = x1
              else:
               print(x1)
               
               print(verts[0][0])
               #ax.add_line(lines.Line2D([0,vert[0]],[310,verts[1]]))
               print(verts[0][0] - x1)
               print("aaaa")

              #if vert[0] > chk:
             
             i+=1
             """

            #ax.add_line(lines.Line2D([verts[n7][0],verts[n9][0]],[verts[n7][1],verts[n9][1]]))
            """
            print("1")
            print(verts[n1][0])
            print(verts[n1][1])
            print("2")
            print(verts[n9][0])
            print(verts[n9][1])
            print("3")
            print(verts[n7][0])
            print(verts[n7][1])
            print("4")
            print(verts[0][0])
            print(verts[0][1])
            """
            """"
            p1_1 = verts[n1][0]
            p1_2 = verts[n1][1]
             
            p2_1 = verts[n9][0]
            p2_2 = verts[n9][1]
             
            p3_1 = verts[n7][0]
            p3_2 = verts[n7][1]
             
            p4_1 = verts[0][0]
            p4_2 = verts[0][1]
            
            point = [p1_1,p1_2,p2_1,p2_2,p3_1,p3_2,p4_1,p4_2]
            """
            
            #ax.add_line(lines.Line2D([verts[n3][0],verts[n8][0]],[verts[n3][1],verts[n8][1]])) 
            
            """
            print(verts[n3][0])
            print(verts[n8][0])
            print(verts[n3][1])
            print(verts[n8][1])
            """

            #print(verts[n8][0] - verts[n3][0])
            #print(verts[n8][1] - verts[n3][1]) 
            
           
            #ax.add_line(lines.Line2D([verts[n2][0],verts[n6][0]],[verts[n2][1],verts[n6][1]])) 
            
            #ax.add_line(lines.Line2D([verts[n4][0],verts[n5][0]],[verts[n4][1],verts[n5][1]])) 
            
            x = (verts[n1][0] + verts[0][0])//2
            y = (verts[0][1] + verts[n1][1])//2

            """
            ax.add_line(lines.Line2D([x,x+200], [y, y])) 
            """
            print("x "+str(x)+" y "+str(y))
            
            plt.scatter(x,y)
            
            angle = get_theta(verts[n1][0]-x,verts[n1][1]-y)
            
            print("angle "+str(i+1)+" "+str(angle))
            # ตรวจจับความเอียงของวัตถุ
            if angle == 90.0:
              print(angle)
              nn = 10
              angle1 = get_theta(verts[nn][0] - verts[0][0],verts[nn][1] - verts[0][1])
              #ax.add_line(lines.Line2D([verts[nn][0],verts[0][0]],[verts[nn][1],verts[0][1]]))
              print("angle1 "+str(angle1))
              
              angle1_ = 180 - angle1
              
              angle2 = get_theta(verts[n-nn][0] - verts[0][0],verts[n-nn][1] - verts[0][1])
              print("angle2 "+str(angle2))
           
              if angle1_ > angle2:
               angle = 91
              else:
               angle = 89
            
            if angle > 90:
    
             nn = (n*80)//100
             
             result_angle = get_theta(verts[n1][0] - verts[nn][0],verts[n1][1] - verts[nn][1])
             #ax.add_line(lines.Line2D([verts[n1][0],verts[nn][0]],[verts[n1][1],verts[nn][1]]))
             if result_angle < 165:
              nn =8
              angle_ = get_theta(verts[nn][0] - verts[0][0],verts[nn][1] - verts[0][1])
              print("angle_"+str(angle_))

              #ax.add_line(lines.Line2D([verts[nn][0],verts[0][0]],[verts[nn][1],verts[0][1]]))
         
              if angle_ > 160 :
               result_angle = angle_ - 90
             
             """
             endx_angle1 =  half_height*math.cos((180-result_angle) * radian);
             endy_angle1 =  half_height*math.sin((180-result_angle) * radian);

             endx1 = x - endx_angle1
             endy1 = y - endy_angle1 

             ax.add_line(lines.Line2D([x,endx1],[y,endy1]))  

             result_angle1 = 90-result_angle

             endx_angle2 =  half_width*math.cos(result_angle1 * radian);
             endy_angle2 =  half_width*math.sin(result_angle1 * radian);

             endx2 = x - endx_angle2
             endy2 = y - endy_angle2

             ax.add_line(lines.Line2D([x,endx2],[y,endy2]))  
             
             nn1 = (n*65)//100 

             #ax.add_line(lines.Line2D([x,verts[nn1][0]],[y,verts[nn1][1]]))
             
             nn2 = (n*40)//100
             print("verts")
             print(len(verts))
             """
             #ax.add_line(lines.Line2D([x,verts[nn2+10][0]],[y,verts[nn2+10][1]]))  

             #endx_angle1 =  obj_height*math.cos((180-result_angle) * radian);
             #endy_angle1 =  obj_height*math.sin((180-result_angle) * radian);

             #endx1 = verts[0][0] - endx_angle1
             #endy1 = verts[0][1] - endy_angle1 

             #ax.add_line(lines.Line2D([verts[0][0],endx1],[verts[0][1],endy1]))  

             #result_angle1 = 90-result_angle

             #endx_angle2 =  obj_width*math.cos(result_angle1 * radian);
             #endy_angle2 =  obj_width*math.sin(result_angle1 * radian);

             #endx2 = endx1 + endx_angle2
             #endy2 = endy1 + endy_angle2

             #ax.add_line(lines.Line2D([endx1,endx2],[endy1,endy2]))  
             
             #endx3 = endx2+endx_angle1
             #endy3 = endy2+endy_angle1

             #ax.add_line(lines.Line2D([endx2,endx3],[endy2,endy3])) 
            
             #ax.add_line(lines.Line2D([endx3,verts[0][0]],[endy3,verts[0][1]]))   
             
             #endx_angle1 =  half_height*math.cos((180-result_angle) * radian);
             #endy_angle1 =  half_height*math.sin((180-result_angle) * radian);

             #endx4 = verts[0][0] - endx_angle1
             #endy4 = verts[0][1] - endy_angle1 
             
             #endx5 = endx2+endx_angle1
             #endy5 = endy2+endy_angle1

             #ax.add_line(lines.Line2D([endx4,endx5],[endy4,endy5]))  

             #result_angle1 = 90-result_angle

             #endx_angle2 =  half_width*math.cos(result_angle1 * radian);
             #endy_angle2 =  half_width*math.sin(result_angle1 * radian);

             #endx2 = endx1 + endx_angle2
             #endy2 = endy1 + endy_angle2
      
             #endx3 = endx3-endx_angle2
             #endy3 = endy3-endy_angle2

             #ax.add_line(lines.Line2D([endx2,endx3],[endy2,endy3]))  
             
            elif angle < 90:
             nn = (n*80)//100

             result_angle = get_theta(verts[nn][0]-verts[n-1][0],verts[nn][1]-verts[n-1][1])
             #ax.add_line(lines.Line2D([verts[nn][0],verts[n-1][0]],[verts[nn][1],verts[n-1][1]]))
             if result_angle < 165:
              nn = 8
              angle_ = get_theta(verts[nn][0] - verts[0][0],verts[nn][1] - verts[0][1])
              print("angle_"+str(angle_))

              #ax.add_line(lines.Line2D([verts[0][0],verts[nn][0]],[verts[0][1],verts[nn][1]]))
         
              if angle_ > 160:
               result_angle = angle_ - 45
            
             #nn1 = (n*60)//100

             #ax.add_line(lines.Line2D([x,verts[nn1][0]],[y,verts[nn1][1]]))
             
             #nn2 = (n*33)//100
             
             #ax.add_line(lines.Line2D([x,verts[nn2+11][0]],[y,verts[nn2+11][1]]))  
             #nnn = (n*80)//100

             #ax.add_line(lines.Line2D([verts[nnn][0],verts[nn2][0]],[verts[nnn][1],verts[nn2][1]]))  
             
             #angle_x1 =  obj_width*math.cos((90-result_angle) * radian);
             #angle_y1 =  obj_width*math.sin((90-result_angle) * radian);

             #print(angle_x1)
             #print(angle_y1)

             #x1 = verts[0][0] - angle_x1
             #y1 = verts[0][1] - angle_y1 

             #ax.add_line(lines.Line2D([verts[0][0],x1],[verts[0][1],y1]))  

             #result_angle1 = 180-result_angle

             #angle_x2 =  obj_height*math.cos(result_angle1 * radian);
             #angle_y2 =  obj_height*math.sin(result_angle1 * radian);

             #x2 = x1 - angle_x2
             #y2 = y1 - angle_y2

             #ax.add_line(lines.Line2D([x1,x2],[y1,y2]))  

             #x3 = x2+angle_x1
             #y3 = y2+angle_y1

             #ax.add_line(lines.Line2D([x2,x3],[y2,y3])) 
             
             #ax.add_line(lines.Line2D([x3,verts[0][0]],[y3,verts[0][1]]))   
             
             #angle_x3 =  half_width*math.cos((90-result_angle) * radian);
             #angle_y3 =  half_width*math.sin((90-result_angle) * radian);

             #x4 = verts[0][0] - angle_x3
             #y4 = verts[0][1] - angle_y3
             
             #x5 = x2+angle_x3
             #y5 = y2+angle_y3
             #ax.add_line(lines.Line2D([x4,x5],[y4,y5]))  

             #angle_x4 =  half_height*math.cos(result_angle1 * radian);
             #angle_y4 =  half_height*math.sin(result_angle1 * radian);

             #x6 = x1 - angle_x4
             #y6 = y1 - angle_y4
      
             #x7 = x3+angle_x4
             #y7 = y3+angle_y4

             #ax.add_line(lines.Line2D([x6,x7],[y6,y7])) 
            
            draw(x,y,result_angle,'width',ax)

            draw(x,y,result_angle,'height',ax)
            print("result "+str(result_angle))
            
           
            """
            theta = np.linspace(0, 2*np.pi,100)

            r = np.sqrt(9000.0) # circle radius

            x1 = r * np.cos(theta) + x
            x2 = r * np.sin(theta) + y
           
            #ax.add_line(lines.Line2D([x,x1], [y, x2])) 
            plt.plot(x1, x2, color='gray')
            """
            #[415:317]

            #[450:0]
            
            p = Polygon(verts, facecolor="none", edgecolor=color,alpha=0.9,closed=True,fill=True)
            
            ax.add_patch(p)
            #point = contours
            attr = {"point":[x,y] ,"angle":result_angle}
            
            attrs.append(attr)
            
    ax.imshow(masked_image.astype(np.uint8))
     
    if detect:
        plt.close()
        return canvas
    # To transform the drawn figure into ndarray X
    fig.canvas.draw()
    X = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    
    X = X.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    # open cv's RGB style: BGR
    if not real_time:
        X = X[..., ::-1]
    if making_video or real_time:
        #cv2.imwrite('splash.png', X)
        plt.close()
        return X,attrs
    elif making_image:
        cv2.imwrite('splash.png', X)
    if auto_show:
       print("aaaaaaaaaaaaaaaaaaa")
       dummy = plt.figure()
       new_manager = dummy.canvas.manager
       new_manager.canvas.figure = fig
       fig.set_canvas(new_manager.canvas)
       fig.show()

def display_differences(image,
                        gt_box, gt_class_id, gt_mask,
                        pred_box, pred_class_id, pred_score, pred_mask,
                        class_names, title="", ax=None,
                        show_mask=True, show_box=True,
                        iou_threshold=0.5, score_threshold=0.5):
    """Display ground truth and prediction instances on the same image."""
    # Match predictions to ground truth
    gt_match, pred_match, overlaps = utils.compute_matches(
        gt_box, gt_class_id, gt_mask,
        pred_box, pred_class_id, pred_score, pred_mask,
        iou_threshold=iou_threshold, score_threshold=score_threshold)
    # Ground truth = green. Predictions = red
    colors = [(0, 1, 0, .8)] * len(gt_match)\
           + [(1, 0, 0, 1)] * len(pred_match)
    # Concatenate GT and predictions
    class_ids = np.concatenate([gt_class_id, pred_class_id])
    scores = np.concatenate([np.zeros([len(gt_match)]), pred_score])
    boxes = np.concatenate([gt_box, pred_box])
    masks = np.concatenate([gt_mask, pred_mask], axis=-1)
    # Captions per instance show score/IoU
    captions = ["" for m in gt_match] + ["{:.2f} / {:.2f}".format(
        pred_score[i],
        (overlaps[i, int(pred_match[i])]
            if pred_match[i] > -1 else overlaps[i].max()))
            for i in range(len(pred_match))]
    # Set title if not provided
    title = title or "Ground Truth and Detections\n GT=green, pred=red, captions: score/IoU"
    # Display
    display_instances(
        image,
        boxes, masks, class_ids,
        class_names, scores, ax=ax,
        show_bbox=show_box, show_mask=show_mask,
        colors=colors, captions=captions,
        title=title)


def draw_rois(image, rois, refined_rois, mask, class_ids, class_names, limit=10):
    """
    anchors: [n, (y1, x1, y2, x2)] list of anchors in image coordinates.
    proposals: [n, 4] the same anchors but refined to fit objects better.
    """
    masked_image = image.copy()

    # Pick random anchors in case there are too many.
    ids = np.arange(rois.shape[0], dtype=np.int32)
    ids = np.random.choice(
        ids, limit, replace=False) if ids.shape[0] > limit else ids

    fig, ax = plt.subplots(1, figsize=(12, 12))
    if rois.shape[0] > limit:
        plt.title("Showing {} random ROIs out of {}".format(
            len(ids), rois.shape[0]))
    else:
        plt.title("{} ROIs".format(len(ids)))

    # Show area outside image boundaries.
    ax.set_ylim(image.shape[0] + 20, -20)
    ax.set_xlim(-50, image.shape[1] + 20)
    ax.axis('off')

    for i, id in enumerate(ids):
        color = np.random.rand(3)
        class_id = class_ids[id]
        # ROI
        y1, x1, y2, x2 = rois[id]
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              edgecolor=color if class_id else "gray",
                              facecolor='none', linestyle="dashed")
        ax.add_patch(p)
        # Refined ROI
        if class_id:
            ry1, rx1, ry2, rx2 = refined_rois[id]
            p = patches.Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, linewidth=2,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)
            # Connect the top-left corners of the anchor and proposal for easy visualization
            ax.add_line(lines.Line2D([x1, rx1], [y1, ry1], color=color))

            # Label
            label = class_names[class_id]
            ax.text(rx1, ry1 + 8, "{}".format(label),
                    color='w', size=11, backgroundcolor="none")

            # Mask
            m = utils.unmold_mask(mask[id], rois[id]
                                  [:4].astype(np.int32), image.shape)
            masked_image = apply_mask(masked_image, m, color)

    ax.imshow(masked_image)

    # Print stats
    print("Positive ROIs: ", class_ids[class_ids > 0].shape[0])
    print("Negative ROIs: ", class_ids[class_ids == 0].shape[0])
    print("Positive Ratio: {:.2f}".format(
        class_ids[class_ids > 0].shape[0] / class_ids.shape[0]))


# TODO: Replace with matplotlib equivalent?
def draw_box(image, box, color):
    """Draw 3-pixel width bounding boxes on the given image array.
    color: list of 3 int values for RGB.
    """
    y1, x1, y2, x2 = box
    image[y1:y1 + 2, x1:x2] = color
    image[y2:y2 + 2, x1:x2] = color
    image[y1:y2, x1:x1 + 2] = color
    image[y1:y2, x2:x2 + 2] = color
    return image


def display_top_masks(image, mask, class_ids, class_names, limit=4):
    """Display the given image and the top few class masks."""
    to_display = []
    titles = []
    to_display.append(image)
    titles.append("H x W={}x{}".format(image.shape[0], image.shape[1]))
    # Pick top prominent classes in this image
    unique_class_ids = np.unique(class_ids)
    mask_area = [np.sum(mask[:, :, np.where(class_ids == i)[0]])
                 for i in unique_class_ids]
    top_ids = [v[0] for v in sorted(zip(unique_class_ids, mask_area),
                                    key=lambda r: r[1], reverse=True) if v[1] > 0]
    # Generate images and titles
    for i in range(limit):
        class_id = top_ids[i] if i < len(top_ids) else -1
        # Pull masks of instances belonging to the same class.
        m = mask[:, :, np.where(class_ids == class_id)[0]]
        m = np.sum(m * np.arange(1, m.shape[-1] + 1), -1)
        to_display.append(m)
        titles.append(class_names[class_id] if class_id != -1 else "-")
    display_images(to_display, titles=titles, cols=limit + 1, cmap="Blues_r")


def plot_precision_recall(AP, precisions, recalls):
    """Draw the precision-recall curve.

    AP: Average precision at IoU >= 0.5
    precisions: list of precision values
    recalls: list of recall values
    """
    # Plot the Precision-Recall curve
    _, ax = plt.subplots(1)
    ax.set_title("Precision-Recall Curve. AP@50 = {:.3f}".format(AP))
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, 1.1)
    _ = ax.plot(recalls, precisions)


def plot_overlaps(gt_class_ids, pred_class_ids, pred_scores,
                  overlaps, class_names, threshold=0.5):
    """Draw a grid showing how ground truth objects are classified.
    gt_class_ids: [N] int. Ground truth class IDs
    pred_class_id: [N] int. Predicted class IDs
    pred_scores: [N] float. The probability scores of predicted classes
    overlaps: [pred_boxes, gt_boxes] IoU overlaps of predictins and GT boxes.
    class_names: list of all class names in the dataset
    threshold: Float. The prediction probability required to predict a class
    """
    gt_class_ids = gt_class_ids[gt_class_ids != 0]
    pred_class_ids = pred_class_ids[pred_class_ids != 0]

    plt.figure(figsize=(12, 10))
    plt.imshow(overlaps, interpolation='nearest', cmap=plt.cm.Blues)
    plt.yticks(np.arange(len(pred_class_ids)),
               ["{} ({:.2f})".format(class_names[int(id)], pred_scores[i])
                for i, id in enumerate(pred_class_ids)])
    plt.xticks(np.arange(len(gt_class_ids)),
               [class_names[int(id)] for id in gt_class_ids], rotation=90)

    thresh = overlaps.max() / 2.
    for i, j in itertools.product(range(overlaps.shape[0]),
                                  range(overlaps.shape[1])):
        text = ""
        if overlaps[i, j] > threshold:
            text = "match" if gt_class_ids[j] == pred_class_ids[i] else "wrong"
        color = ("white" if overlaps[i, j] > thresh
                 else "black" if overlaps[i, j] > 0
                 else "grey")
        plt.text(j, i, "{:.3f}\n{}".format(overlaps[i, j], text),
                 horizontalalignment="center", verticalalignment="center",
                 fontsize=9, color=color)

    plt.tight_layout()
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")


def draw_boxes(image, boxes=None, refined_boxes=None,
               masks=None, captions=None, visibilities=None,
               title="", ax=None):
    """Draw bounding boxes and segmentation masks with differnt
    customizations.

    boxes: [N, (y1, x1, y2, x2, class_id)] in image coordinates.
    refined_boxes: Like boxes, but draw with solid lines to show
        that they're the result of refining 'boxes'.
    masks: [N, height, width]
    captions: List of N titles to display on each box
    visibilities: (optional) List of values of 0, 1, or 2. Determine how
        prominant each bounding box should be.
    title: An optional title to show over the image
    ax: (optional) Matplotlib axis to draw on.
    """
    # Number of boxes
    assert boxes is not None or refined_boxes is not None
    N = boxes.shape[0] if boxes is not None else refined_boxes.shape[0]

    # Matplotlib Axis
    if not ax:
        _, ax = plt.subplots(1, figsize=(12, 12))

    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    margin = image.shape[0] // 10
    ax.set_ylim(image.shape[0] + margin, -margin)
    ax.set_xlim(-margin, image.shape[1] + margin)
    ax.axis('off')

    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        # Box visibility
        visibility = visibilities[i] if visibilities is not None else 1
        if visibility == 0:
            color = "gray"
            style = "dotted"
            alpha = 0.5
        elif visibility == 1:
            color = colors[i]
            style = "dotted"
            alpha = 1
        elif visibility == 2:
            color = colors[i]
            style = "solid"
            alpha = 1

        # Boxes
        if boxes is not None:
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=alpha, linestyle=style,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Refined boxes
        if refined_boxes is not None and visibility > 0:
            ry1, rx1, ry2, rx2 = refined_boxes[i].astype(np.int32)
            p = patches.Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, linewidth=2,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)
            # Connect the top-left corners of the anchor and proposal
            if boxes is not None:
                ax.add_line(lines.Line2D([x1, rx1], [y1, ry1], color=color))

        # Captions
        if captions is not None:
            caption = captions[i]
            # If there are refined boxes, display captions on them
            if refined_boxes is not None:
                y1, x1, y2, x2 = ry1, rx1, ry2, rx2
            x = random.randint(x1, (x1 + x2) // 2)
            ax.text(x1, y1, caption, size=11, verticalalignment='top',
                    color='w', backgroundcolor="none",
                    bbox={'facecolor': color, 'alpha': 0.5,
                          'pad': 2, 'edgecolor': 'none'})

        # Masks
        if masks is not None:
            mask = masks[:, :, i]
            masked_image = apply_mask(masked_image, mask, color)
            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                #p = Polygon(verts)#, facecolor="none", edgecolor=[1,0,0,0],fill=1)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))


def display_table(table):
    """Display values in a table format.
    table: an iterable of rows, and each row is an iterable of values.
    """
    html = ""
    for row in table:
        row_html = ""
        for col in row:
            row_html += "<td>{:40}</td>".format(str(col))
        html += "<tr>" + row_html + "</tr>"
    html = "<table>" + html + "</table>"
    IPython.display.display(IPython.display.HTML(html))


def display_weight_stats(model):
    """Scans all the weights in the model and returns a list of tuples
    that contain stats about each weight.
    """
    layers = model.get_trainable_layers()
    table = [["WEIGHT NAME", "SHAPE", "MIN", "MAX", "STD"]]
    for l in layers:
        weight_values = l.get_weights()  # list of Numpy arrays
        weight_tensors = l.weights  # list of TF tensors
        for i, w in enumerate(weight_values):
            weight_name = weight_tensors[i].name
            # Detect problematic layers. Exclude biases of conv layers.
            alert = ""
            if w.min() == w.max() and not (l.__class__.__name__ == "Conv2D" and i == 1):
                alert += "<span style='color:red'>*** dead?</span>"
            if np.abs(w.min()) > 1000 or np.abs(w.max()) > 1000:
                alert += "<span style='color:red'>*** Overflow?</span>"
            # Add row
            table.append([
                weight_name + alert,
                str(w.shape),
                "{:+9.4f}".format(w.min()),
                "{:+10.4f}".format(w.max()),
                "{:+9.4f}".format(w.std()),
            ])
    display_table(table)
