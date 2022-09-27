# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 14:39:08 2022

@author: mitran
"""


import numpy as np
import cv2
import time

def covmat(x,y):
    return np.matmul(np.matmul(x,y),x.T)

class KalmanFilter2D(object):
    def __init__(self,initial_state,dt,acc_std,meas_std):
        
        self.dt = dt
        self.Xn = initial_state
        
        
        self.F = np.array([[1,dt,0,0],
                           [0,1,0,0],
                           [0,0,1,dt],
                           [0,0,0,1]])
        
        self.G = np.array([[0.5 * dt**2 , 0],
                          [dt , 0],
                          [0 , 0.5 * dt**2],
                          [0 , dt]])
        
        
        # choose the tracking parameter among the internal states (Eg position)
        self.H = np.array([[1,0,0,0],
                           [0,0,1,0]])
        
        self.P = np.eye(self.F.shape[1]) * 1000
        self.I = np.eye(self.H.shape[1])
        
        # Q noise matrix : compute prediction variance for the states based on a noisy parameter(controls) like accelerator
        self.Q = self.G @ acc_std**2 @ self.G.T 
        print(self.Q)
        
        [meas_std_x,mean_std_y] = meas_std 
        
        self.R =  np.array([[meas_std_x**2,0],[0,mean_std_y**2]])
    
  
    
    def UpdateEstimate(self,Zn):
        # Zn : Measurement vector
        self.Xn = self.Xn + self.KG @ (Zn - np.matmul(self.H , self.Xn))
        
    def StatePrediction(self,Un):
        
        # depends on the Problem statement Dynamics
        # different for each problem ( dynmaics equations are generalized using Linear algebra)
        
        # Un is the controls for system if its not used make it zero
        self.Xn =np.matmul(self.F,self.Xn) + np.matmul(self.G,Un)
        
        return self.H @ self.Xn  
    
    def KalmanGain(self):
        
        # innovation : Z - Hx        
        Innovation_variance = self.R + covmat(self.H,self.P)
                
        self.KG = np.matmul(self.P @ self.H.T,np.linalg.inv(Innovation_variance))
        
    
    def ProcessUncertainity(self):
        
        # noise matrix Q will be based on the variance of (noisy parameters(like acceleration) and their impacted parameters) 
        
        self.P = covmat(self.F,self.P) + self.Q
        
    def CovarianceUpdate(self):
        # if KG is low estimate will be in a near place and estimate certainity will converge slow
        # else measurements are having low uncertainity and estimate will move towards the measurement and it would converge faster
        self.P = (self.I - self.KG @ self.H) @ self.P @  (self.I - self.KG @ self.H).T
        


def DetectBall(image):
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny_image = cv2.Canny(image,  50, 190, 3)
    ret, img_thresh = cv2.threshold(canny_image, 210, 255,cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0 :
        return [0,0],None
    
    circle = max([cv2.minEnclosingCircle(c) for c in contours],key = lambda x : x[1])
    
    if circle[1]<7:
        return [0,0],None
    return np.array(circle[0],dtype=np.int32),int(circle[1])+1 # circle[0] coordinates of the circle



VideoCap = cv2.VideoCapture('./randomball.avi')


dt = 0.1
acc_std = np.eye(2)*1.01
meas_std = [0.5,0.5]
acceleration_control= np.array([1.,1.]).reshape(-1,1)

initial_state = np.array([0,0,0,0]).reshape(-1,1)

model = KalmanFilter2D(initial_state, dt, acc_std,meas_std)




ret = True


while(ret):
    
    ret, frame = VideoCap.read()
    [x,y] , radius = DetectBall(frame)
    
    Z = np.array([x,y]).reshape(-1,1)
    track = np.array([x,y]).reshape(-1,1)
    Z = Z + np.random.normal(0, 40, size = Z.shape) # Noise
    
    
    if(radius):
        
        
        
        pred = model.StatePrediction(acceleration_control)

        model.ProcessUncertainity()

        model.KalmanGain()

        model.UpdateEstimate(Z)
         
        model.CovarianceUpdate()  
        
        frame = cv2.rectangle(frame, (x - radius, y - radius), (x + radius, y + radius), (255, 0, 0), 2)
        frame = cv2.rectangle(frame, ( int(pred[0] - radius), int(pred[1] - radius) ), ( int(pred[0] + radius), int(pred[1] + radius) ), (255, 255, 0), 2)
        frame = cv2.rectangle(frame, ( int(Z[0] - radius), int(Z[1] - radius) ), ( int(Z[0] + radius), int(Z[1] + radius) ), (0, 0, 255), 2)
        
    cv2.imshow('image', frame)
    time.sleep(0.5)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        VideoCap.release()
        cv2.destroyAllWindows()
        break
    












ret, frame = VideoCap.read()
[x,y] , radius = DetectBall(frame)


 


















































































