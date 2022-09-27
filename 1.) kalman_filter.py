# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 10:15:00 2022

@author: mitran
"""

import numpy as np

def covmat(x,y):
    return np.matmul(np.matmul(x,y),x.T)

class KalmanFilter(object):
    def __init__(self,initial_state,dt,acc_std,meas_std):
        
        self.dt = dt
        self.Xn = initial_state
        
        
        self.F = np.array([[1,dt],[0,1]])
        self.G = np.array([0.5 * dt**2 , dt]).reshape(-1,1)
        
        
        # choose the tracking parameter among the internal states (Eg position)
        self.H = np.array([[1,0]])
        self.P = np.eye(self.F.shape[1])
        self.I = np.eye(self.F.shape[1])
        
        # Q noise matrix : compute prediction variance for the states based on a noisy parameter(controls) like accelerator
        
        self.Q = self.G @ np.array([acc_std**2]).reshape(1,1) @ self.G.T 
        print(self.Q)
        
        self.R =  np.array([meas_std**2])
    
  
    
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
        print(self.KG)
        
    
    def ProcessUncertainity(self):
        
        # noise matrix Q will be based on the variance of (noisy parameters(like acceleration) and their impacted parameters) 
        
        self.P = covmat(self.F,self.P) + self.Q
        
    def CovarianceUpdate(self):
        # if KG is low estimate will be in a near place and estimate certainity will converge slow
        # else measurements are having low uncertainity and estimate will move towards the measurement and it would converge faster
        
        self.P = (self.I - self.KG) @ self.P
        
        
dt = 0.1
T = np.arange(0, 200, dt)
Track = 0.001*((T**3) - T)
acc_std = 0.25
meas_std = 1

initial_state = np.array([0,0]).reshape(-1,1)

model = KalmanFilter(initial_state, dt, acc_std,meas_std)

measurement = []
predictions = []

acceleration_control = np.array([2]).reshape(-1,1)

for x in Track:
    
    
    z = x + np.random.normal(0, 500)
    z = z.reshape(1,1)
    
    
    measurement.append(z.squeeze(1))
    
    # Prediction
    
    pred = model.StatePrediction(acceleration_control) # constant acceleration
    predictions.append(pred.squeeze(1))
    
    model.ProcessUncertainity()
    
    # Update
    
    model.KalmanGain()
    model.UpdateEstimate(z)
    
    model.CovarianceUpdate()   
    
    print(x,z)
    

import matplotlib.pyplot as plt      
        
fig = plt.figure()
fig.suptitle('Kalman filter in 1-D', fontsize=20)
plt.plot(T, measurement, label='Measurements', color='b',linewidth=0.5)
plt.plot(T, np.array(Track), label='Real Track', color='black', linewidth=1.5)
plt.plot(T, np.squeeze(predictions), label='Kalman Filter Prediction', color='r', linewidth=1.5)
plt.xlabel('Time (s)', fontsize=20)
plt.ylabel('Position (m)', fontsize=20)
plt.legend()
plt.show()  
