# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 11:08:53 2022

@author: mitran
"""

import numpy as np

# 1D Markov Localisation using Bayes Filter : Udacity

mapsize = 25
U = 1 # Control
Q = 1 # Control std deviation
R = 1 # Observation std deviation

landmark_map = [3, 9, 14, 23]

SensorReadings =  [[1,7,12,21], [0,6,11,20], [5,10,19], [4,9,18],
                                    [3,8,17], [2,7,16], [1,6,15], [0,5,14], [4,13],
                                    [2,11],[1,10],[8],[7],[6],[5],[4],[2],[1],[0], 
                                    ]
T = len(SensorReadings)
# Initialize priors

# The initial belief of the states(0 to map size) will be based on landmark location and deviation
# there will be a peak and some deviation in the lamdmark positions
priors = [0] * mapsize # initial belief
Posteriors = [0] * mapsize

norm = len(landmark_map) * ( (2*R)+1 )

for l in landmark_map:
    
    priors[l-1] = 1/norm
    priors[l+1] = 1/norm
    priors[l] = 1/norm
    
def gaussian(x,mu,std):
   return 1./(np.sqrt(2.*np.pi)*std)*np.exp(-np.power((x - mu)/std, 2.)/2)
def MotionModel(Xt,Ut,size):
    """ To Compute total probability for Xt(pseudo position)
    The system moves from some position in the map to Xt with Ut control in controld variance
    solving this p(xt​∣xt−1(i)​,ut​,m) ∗ bel(xt−1(i)​) for all possible Xt-1 in the environment and summing up to find total probability
    """
    total_prob = 0 # if its continous will be using integeral formula
    for i in range(mapsize):
        
        prev_pos= float(i)
        
        dist = Xt - prev_pos        
        trans_prob = gaussian(dist, Ut, Q)
        bel_x_1 = priors[i]
        
        total_prob += trans_prob * bel_x_1
        
    return total_prob

def ObservationModel(Xt,Z):
    """

    Given the observation from sensor .
    we need to compute P(Z|Xt) since Z is a vector of k observations
    P( [Z1,Z2....ZK|Xt] ) which is PROD i (1->k) P(Zi|Xt)   

    """
    Z = sorted(Z)
    pseudo_ranges = [] # ranges from the Xt to all landmarks 
    
    for L in landmark_map:
        if(L<Xt):
            pass
        else:
            pseudo_ranges.append(L-Xt)
    pseudo_ranges.sort()
    
    #print(pseudo_ranges)
    
    if(len(pseudo_ranges)<len(Z)) : return 0
        
    P_Z_xt = 1.0
    
    for i in range(len(Z)):
        
        zi = Z[i]
        P_zi_Xt = gaussian(zi,pseudo_ranges[i],R)
        
        P_Z_xt *= P_zi_Xt
        
    return P_Z_xt

def normalize(X):
    # sum of the vector is one
    sumX = sum(X)
    return [i/sumX for i in X]
def maxargmax(X):
    val = [0,X[0]]
    for i,x in enumerate(X):
        if(x>val[1]):
            val = [i,x]
    return val
  

for t in range(T):
    
    if(len(SensorReadings[t])):
        obs = SensorReadings[t]
    else:
        obs = [float(mapsize)]
    
    
    # iterate the pseudo position Xt
    for Xt in range(mapsize):
        
        pseudo_pos = float(Xt)
        
        # Motion Model : p(xt​∣xt−1(i)​,ut​,m) ∗ bel(xt−1(i)​)
        motion_prob = MotionModel(Xt,U,mapsize)
        
        # Obsevation Model
        obs_prob = ObservationModel(Xt,SensorReadings[t])
        
        Posteriors[Xt] = motion_prob * obs_prob
        
    Posteriors = normalize(Posteriors)
    print(maxargmax(Posteriors))
    for i in range(mapsize):
        priors[i] = Posteriors[i]
        
        
        
    
    