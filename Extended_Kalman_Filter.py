# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 09:06:10 2022

@author: mitran
"""
import numpy as np
import math
from abc import ABCMeta, abstractmethod
import pandas as pd
import cv2
import time


def covmat(x,y):
    return np.matmul(np.matmul(x,y),x.T)

def DrawPosition(Image,x,y,radius,angle,color=(0,255,0)):
    
    #circle
    Image = cv2.circle(Image, (x,y), radius, color, 1)
    #line
    coords = ( math.cos( math.radians(angle) ) * radius  , math.sin( math.radians(angle) ) * radius )
    Image = cv2.line(Image, (x,y),(x + int(coords[0]) , y + int(coords[1])  ) , color, 1) 
    
    return Image

    

class ExtendedKalmanFilter(object):
    
    
    __metaclass__ = ABCMeta
    
    def __init__(self,initial_state,dt,control_std,meas_std):
        
        self.dt = dt
        self.Xn = initial_state
        
        
        #self.F = Handled by Function method
        
        #self.G = Handled by Function method
        
        self.P = np.eye(initial_state.shape[0]) * 1000
        self.I = np.eye(initial_state.shape[0])
        
        #self.Q = Handled by setF method
        
        [meas_std_x,mean_std_y,mean_std_a] = meas_std 
        self.R =  np.array([ [meas_std_x**2,0,0],[0,mean_std_y**2,0],[0,0,mean_std_a**2] ])
        
        [cv_v,cv_w] = control_std
        self.ControlVariance =  np.array([ [cv_v**2,0],[0,cv_w**2] ])        
        
        
        #self.H = Handled by HFunction method
        self.H = np.eye(initial_state.shape[0])
        
        
       

    
    def UpdateEstimate(self,Zn):
        # Zn : Measurement vector
        # make the below nonlinear
        # converting state to measurent structure can be non linear when sensor in nonlinear(eg:gps,imu,radar)
        # so replace it to a function
        self.Xn = self.Xn + self.KG @ (Zn - np.matmul(self.H , self.Xn)) # change
        # The child class is forced to impleemt the function method
        #self.Xn = self.Xn + self.KG @ (Zn - self.Hfunction(self.Xn))
        
        
    def StatePrediction(self,Un):
        
        # since is non linear filter state prediction and control are made to a non linear function
        """self.Xn =np.matmul(self.F,self.Xn) + np.matmul(self.G,Un)"""
        # The child class is forced to impleemt the function method
        self.Xn = self.Function(self.Xn,Un)
        
        return self.Xn  
    
    def KalmanGain(self):
        
        # innovation : Z - Hx        
        # since variance cannot pass through non linear Hfunction self.H is Jacobian of non linear Hfunction
        Innovation_variance = self.R + covmat(self.H,self.P)
                
        self.KG = np.matmul(self.P @ self.H.T,np.linalg.inv(Innovation_variance))
        
    
    def ProcessUncertainity(self):
        
        # noise matrix Q will be based on the variance of (noisy parameters(like acceleration) and their impacted parameters) 
        # since variance cannot pass through non linear function self.F is Jacobian of non linear function
        self.P = covmat(self.F,self.P) + self.Q
        
    def CovarianceUpdate(self):
        # if KG is low estimate will be in a near place and estimate certainity will converge slow
        # else measurements are having low uncertainity and estimate will move towards the measurement and it would converge faster
        self.P = (self.I - self.KG @ self.H) @ self.P @  (self.I - self.KG @ self.H).T 
    
    @abstractmethod
    def Function(self, X ,U):
       
        raise NotImplementedError()    
    @abstractmethod
    def HFunction(self, x):
       
        raise NotImplementedError()  
        
    @abstractmethod
    def setF(self, X ,U):
       
        raise NotImplementedError()    
        
    @abstractmethod
    def setH(self, x):
       
        raise NotImplementedError()   
        
class TrackEKF(ExtendedKalmanFilter):
    def __init__(self,**kwargs):
        
        super().__init__(**kwargs)
    
    def Function(self, X,U):
        
        # x : state u:controls
        out = np.zeros_like(X)
        
        out[0] = X[0] + math.cos( math.radians(X[2]) ) * U[0]
        out[1] = X[1] + math.sin( math.radians(X[2]) ) * U[0]
        out[2] = X[2] + U[1]
        
        self.setF(X, U)
        
        return out
    
    def setF(self, X,U):
        
        # since the above function is non linear function variance cannot be computed with that., so we need the linear function of it 
        # Compute Jacobian of the above function ( State, Control )
        
        self.F = np.array( [ [1, 0, -math.sin( math.radians(X[2]) ) * float(U[0])],
                             [0, 1,  math.cos( math.radians(X[2]) ) * float(U[0])],
                             [0, 0, 1],                             
                             ] )
        
        # Jacobian for Controls
        G = np.array([ [math.cos( math.radians(X[2]) ) , 0],
                       [math.sin( math.radians(X[2]) ) , 0],
                       [0, 1]
            ])
        self.Q = G @ self.ControlVariance @ G.T
        
Data = pd.read_csv("./non_linear_dynamics.csv")

vid_capture = cv2.VideoCapture('non_linear_dynamics.avi')

check_length = False
if(check_length):
    imgs = 0
    while(True):
        ret, frame = vid_capture.read()
        if(ret==False):break
        print(imgs)
        imgs+=1
        
    
    assert imgs==len(Data)," video not aligned with dataframe"


dt = 0.1

control_std = [0.25,0.05]
meas_std = [5,5,0.5]

initial_state =  np.array([0,0,0]).reshape(-1,1)

model = TrackEKF(initial_state=initial_state, dt=dt, control_std=control_std, meas_std=meas_std)

measurement = []
predictions = []

vid_capture = cv2.VideoCapture('non_linear_dynamics.avi')




for i in range(len(Data)):
    
    vel,ang_vel,pos_x,pos_y,angle = Data.iloc[i][["velocity","angular_velocity","pos_x","pos_y","angle"]]
    
    controls = np.array([vel,ang_vel]).reshape(-1,1)
    Z = np.array([pos_x,pos_y,angle]).reshape(-1,1)
    
    
    # Prediction
    
    pred = model.StatePrediction(controls) # controls    
    model.ProcessUncertainity()
    
    # Update
    
    model.KalmanGain()
    model.UpdateEstimate(Z)
    
    model.CovarianceUpdate()   
    
    
    
    _, frame = vid_capture.read()
    frame = DrawPosition(frame,int(pred[0][0]),int(pred[1][0]),40,pred[2][0])
    frame = DrawPosition(frame,int(Z[0][0]),int(Z[1][0]),40,Z[2][0],color=(0,0,255))
    
    
    
    time.sleep(0.5)
    cv2.imshow("agent",frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
    
    
    
        
        
    
    
        
        

    

