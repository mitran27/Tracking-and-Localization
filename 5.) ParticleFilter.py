# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 10:27:12 2022

@author: mitran
"""

# Particle Filter

from abc import ABCMeta, abstractmethod
import random,math
import numpy as np
import pandas as pd
import copy
import cv2
import time


WORLD_SIZE = 1000

# Implementing cos function and sin function with radians

def COS(X):
    
    return math.cos( math.radians(X) )


def SIN(X):
    
    return math.sin( math.radians(X) )

def gaussian(x,mu,std):
   gauss = np.exp(-np.power(x - mu, 2.) / (2 * np.power(std, 2.)))
   return gauss

def DrawPosition(Image,x,y,radius,angle,color=(0,255,0)):
    
    #circle
    Image = cv2.circle(Image, (x,y), radius, color, 1)
    #line
    coords = ( math.cos( math.radians(angle) ) * radius  , math.sin( math.radians(angle) ) * radius )
    Image = cv2.line(Image, (x,y),(x + int(coords[0]) , y + int(coords[1])  ) , color, 1) 
    
    return Image
def place_image(back,img,pos):
    h,w,_ = img.shape
    pos = [int(i) for i in pos]
    
    x_offset = pos[0]-w//2
    y_offset = pos[1]-h//2
    
    
    
    back[y_offset:y_offset+h,x_offset:x_offset+w] = img
    return back
    

class NoiseGen():
    def __init__(self,noise):
        self._noise = noise
        
    def __getitem__(self,i):        
        x =  np.random.normal(0,  self._noise[i], 1)[0]
        return x
    
class Particle():
    
    __metaclass__ = ABCMeta
    ID = 1
    
    def __init__(self,weight=1):
        
        
        self.W = weight
        self.id = Particle.ID
        Particle.ID += 1
        
    @abstractmethod
    def set_state(self):
       
        raise NotImplementedError()  
        
    @abstractmethod
    def Move(self):
       
        raise NotImplementedError()  


        
class RobotParticle(Particle):
    
    def __init__(self,weight=1):
        super().__init__(weight=weight)
        self.set_state()
        
    def set_state(self):
        
        # State is different w.r.t Usecases
        self.State = [ random.uniform(20,WORLD_SIZE-20),
                       random.uniform(20,WORLD_SIZE-20),
                       random.uniform(0,360)
            ]
        
        self._Noise = NoiseGen([5,5,1])
    def Move(self,Controls,Noise):
        
        # implements th dynamics function
        
        
        self.State[0] = self.State[0] + COS( self.State[2] ) * Controls[0] 
        self.State[1] = self.State[1] + SIN( self.State[2] ) * Controls[0] 
        self.State[2] = (self.State[2] + Controls[1])%360
        
        if(Noise):
            
            self.State[0] += self._Noise[0] 
            self.State[1] += self._Noise[1]
            self.State[2] += self._Noise[2]
        
        
        
        
            
    @staticmethod   
    def Compute(L,state):
        
        # L is the Landmarks position and state is the particle position
        # COmpute transformed coordinated : compute the landmark coordinate w.r.t particle as base axis(0,0,direction along x axis)

        Xl,Yl = L
        X,Y,A = state
        
        # Translation
        # Move all cordinates to particle as base axis ( so subtract by state)
        
        x = Xl - X
        y = Yl - Y
        D = ( x**2 + y**2 ) ** 0.5


        # Rotation
        
        # Landmarks observation are wrt to yaw angle(A) to make it along x axis rotate anticlockwise A
        
        x_ = x * COS(A) - y * SIN(A)
        y_ = y * COS(A) + x * SIN(A)
        
        # translation and rotation can be done simultaneously with Homogeneous transformation
        
        return x_,y_,D      
        
    
    def Sense(self,Landmarks):
        
        # Compute coordinate relationship between state and the landmarks
        # Considering (angle and distance) bw particle and landmark
        
        Observations = []   
        for L in Landmarks:
            x,y,D = self.Compute(L,self.State)
            # if we need a local window landmarks we can have a condition for D
            Observations.append( (x,y,D) )
        return Observations  
           
    def ComputeOBSProb(self,Oberservation,SensorReading,meas_std):
        
    
        
        
        P_Z_xt = 1.0
        
        for i in range(len(SensorReading)):
            
            zi = SensorReading[i]
            P_zi_Xt = gaussian(zi[0],Oberservation[i][0],meas_std[0]) *  gaussian(zi[1],Oberservation[i][1],meas_std[1])
            P_Z_xt *= P_zi_Xt
        return P_Z_xt
                
        
 
def getLandmarks(count,dims_size):
    
    Landmarks = []
    for i in range(count):
        lm = []
        for d_sze in dims_size:
            lm.append(random.random() * d_sze)
        Landmarks.append(lm)
    return Landmarks
        
    
class ParticleFilter(object):
    
    def __init__(self,N):
        
        self.Particles = [None]*N
        self.N = N
        for i in range(N):
            self.Particles[i] = RobotParticle( weight = float(1) )
    
    def Move(self, Controls, Noise = True): # noise is important bcz the dynamics of the agent is noisy and it has to converge to the real location from assumed location
        for i in range(self.N):
           
            self.Particles[i].Move(Controls,Noise)
    def UpdateWeights(self,Landmarks,SensorReadings,meas_std):
        
        for i in range(self.N):
            
            OBS = self.Particles[i].Sense(Landmarks)
            # if window landmarks association should be done
            
            prob_Z_Xt = self.Particles[i].ComputeOBSProb(OBS,SensorReadings,meas_std) # prob that OBS is Sensor reading with meas_std
            prob_transition = 1 # its apx to 1 bcz the algorithm only moves the particle from the prev state to new state (Known event) so prob of transition from Xt-1 to Xt is 1
            bel_x_1 = self.Particles[i].W
            
            
            # since there is one prev state for a particle no need of total prob
            self.Particles[i].W = prob_Z_Xt * prob_transition * bel_x_1 # since theres is one prev state mul directly
            print(self.Particles[i].W)
    def Normalize(self):
        # Normalize the particle weights
        Wsum = 0.0
        for i in range(self.N):
            Wsum += self.Particles[i].W
            
        for i in range(self.N):
            self.Particles[i].W *= (self.N/Wsum)
            
        
         
    def ReSample(self):
        
        #Roulete Wheel method          
        
        # there are many methods for resampling 
        N= self.N
        rnds = 1/(3*N) # randomness
       
        Wheel = [i + random.uniform(-rnds, rnds) for i in range(1,N+1)]
        Wheel[0] = max(0,Wheel[0])
        Wheel[-1] = min(self.N,Wheel[-1])
        
        pref_sum = 0
        wheel_idx=0
        new_particles = []
        for i in range(self.N):
            
            pref_sum += self.Particles[i].W
            while(wheel_idx<N and Wheel[wheel_idx]<=pref_sum):
                #print(pref_sum)
                new_particles.append(copy.deepcopy(self.Particles[i]))
                wheel_idx+=1
        while(wheel_idx<N):
            new_particles.append(copy.deepcopy(self.Particles[i]))
            wheel_idx+=1
            
        self.Particles = new_particles        
    
            
       
        
        
Landmarks = getLandmarks(15,[WORLD_SIZE,WORLD_SIZE])
Landmarks_img =cv2.resize(cv2.imread("./landmark.jfif"),(10,10))
        
PF = ParticleFilter(1000)  
meas_std = [150,150]
write=True



Data = pd.read_csv("./PF_dynamics.csv")
vid_capture = cv2.VideoCapture('non_linear_dynamics.avi')


for i in range(len(Data)):
    

    vel,ang_vel,pos_x,pos_y,angle = Data.iloc[i][["velocity","angular_velocity","pos_x","pos_y","angle"]]
    
   
    
    state = [ pos_x,pos_y,angle ]
    SensorReadings = [ RobotParticle.Compute(L, state) for L in Landmarks]
    
    
    _, frame = vid_capture.read()
    for particl in PF.Particles:
        par = particl.State
        frame = DrawPosition(frame,int(par[0]),int(par[1]),10,par[2],color=(0,255,255))
    frame = DrawPosition(frame,int(pos_x),int(pos_y),40,angle,color=(255,0,255))
    for land in Landmarks:
        frame = place_image(frame, Landmarks_img, land)
    
    PF.Move([vel,ang_vel]) # vel,ang_vel have some variance hence the moved state is noisy
    PF.UpdateWeights(Landmarks,SensorReadings,meas_std)
    PF.Normalize()
    PF.ReSample()
    
    if(write):
        cv2.imwrite("./Agent/agent"+str(i)+".jpg",frame)
    else:
        cv2.imshow("agent",frame)
        if cv2.waitKey(2) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
  






  
        
