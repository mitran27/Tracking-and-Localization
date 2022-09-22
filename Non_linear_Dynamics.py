# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 12:22:58 2022

@author: mitran
"""

import pandas as pd
import random
import numpy as np
import cv2
import math

import time


def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h), borderValue=(255,255,255))
    return rotated_mat

def place_image(back,img,pos):
    h,w,_ = img.shape
    pos = [int(i) for i in pos]
    
    x_offset = pos[0]-w//2
    y_offset = pos[1]-h//2
    
    
    
    back[y_offset:y_offset+h,x_offset:x_offset+w] = img
    return back


def capture_state(data):
    # sensor captures the location with noise
    data[0]+=np.random.normal(0, 50, 1)[0]
    data[1]+=np.random.normal(0, 50, 1)[0]
    data[2]+=np.random.normal(0, 5, 1)[0]
    
    return data


agent = cv2.imread("agent_img.jpg")
agent = cv2.resize(agent,(20,50))
agent = rotate_image(agent, -90)

out_vid = cv2.VideoWriter('non_linear_dynamics.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (2000,2000))


state = [550,150,0]
v = 7
w = 2

control = 1


dt = 1
DATA = []

for i in range(5000):
    
    Environment = np.ones((2000,2000,3), np.uint8)*255


    if(state[2] <-270):control =1
    if(state[2] >90):control = -1
    curr_v = v + np.random.normal(0, 3, 1)[0]
    curr_w = ( w + np.random.normal(0, 0.5, 1)[0] ) * control
    
    state[0] += math.cos(math.radians(state[2]))*curr_v
    state[1] += math.sin(math.radians(state[2]))*curr_v
    state[2] += curr_w
    
    rot_agent = rotate_image(agent, -state[2])
    imggg = place_image(Environment, rot_agent, state[:2])
    out_vid.write(imggg)
    
    
    
    time.sleep(0.01)
    cv2.imshow("agent",imggg)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
    
    
    
        
    
    data = capture_state(state.copy())
   
    print(i)
    
    DATA.append([v,w*control,*data])
    
    
df = pd.DataFrame(DATA, columns =['velocity', 'angular_velocity', 'pos_x', 'pos_y', 'angle'], dtype = float) 
df.to_csv("non_linear_dynamics.csv")



 
    
out_vid.release()
    

