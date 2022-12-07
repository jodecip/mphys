#!/usr/bin/env python
# coding: utf-8

# In[49]:


import qutip as qt
import numpy as np
import csv
import time
import scipy.integrate as integrate
from matplotlib import pyplot as plt


# In[72]:



with open('negat.csv') as f:
    reader = csv.reader(f)
    r = []
    maxy = []
    max3 = []
    mag = []
    coherencemaxw = []
    coherencemaxwo =[]
    miny = []
    min3 = []
    coherenceminw =[]
    coherenceminwo =[]
    negativitymax = []
    negativitymin =[]
    
    #w stands for with nuclear spin, wo stands for without
    
    for row in reader:
        mag.append(row[0])
        r.append(row[1])
        maxy.append(float(row[2]))
        max3.append(row[3])
        coherencemaxw.append(float(row[4]))
        coherencemaxwo.append(float(row[5]))
        negativitymax.append(float(row[6]))
        negativitymin.append(float(row[7]))
        miny.append(float(row[8]))        
        min3.append(row[9])
        coherenceminw.append(float(row[10]))
        coherenceminwo.append(float(row[11]))
#this sections reads from the text file and saves the relevant columns to variables with suitable names.


# In[73]:


#calculating coherence difference

diffcoh = np.zeros(length)
for i in range(length):
    diffcoh[i] = abs(coherencemaxw[i] - coherenceminw[i])
    
diffcohwo = np.zeros(length)
for i in range(length):
    diffcohwo[i] = abs(coherencemaxwo[i] - coherenceminwo[i])


# In[ ]:


#claculating mean coherence

meancoh = np.zeros(length)
for i in range(length):
    meancoh[i] = abs(coherencemaxw[i] + coherenceminw[i])/2
    
meancohwo = np.zeros(length)
for i in range(length):
    meancohwo[i] = abs(coherencemaxwo[i] + coherenceminwo[i])/2


# In[158]:


#this splits the rows of information into arrays which contain radical pair orientations of the same magnitude
indexes = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
relposition = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
#all possible magntidues of r

for j in relposition:
    for i in range(0,3200):
        if int(mag[i]) == j:
            indexes[j-3].append(i)
            
#print(indexes)


# In[159]:


#this code calculated the compass sensitivity, negativity difference and negativity mean respectively
length = len(maxy) 
compsens = np.zeros(length)
negat = np.zeros(length)
negatmean = np.zeros(length)

for i in range(length):
    compsens[i] = abs(maxy[i] - miny[i])
    negat[i] = abs(negativitymax[i] - negativitymin[i])
    negatmean[i] = abs(negativitymax[i] + negativitymin[i])/2
    
#plt.hist(compsens)
#plt.hist(negat)
#plt.hist(negatmean)


# In[119]:


#this section of code splits the arrays containing negativities, coherences and compass sensitivities into a 2d array where each element in the array contains data for a specific magntiude of r


diffcoherence = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
for i in range(16):
    for j in indexes[i]:
       
        diffcoherence[i].append(diffcohwo[j])
        
compsensivity = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
for i in range(16):
    for j in indexes[i]:
       
        compsensivity[i].append(compsens[j])
        
negativitiesmean = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
for i in range(16):
    for j in indexes[i]:
       
        negativitiesmean[i].append(negatmean[j]) 
        
negativities = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
for i in range(16):
    for j in indexes[i]:
       
        negativities[i].append(negat[j]) 


# In[154]:


#this code scatter plots compass sensitivity vs negativity difference (this can be changed to coherence)

from cycler import cycler
from matplotlib.cm import get_cmap
from matplotlib.pyplot import cm

color = iter(cm.viridis_r(np.linspace(0, 1, 15)))

           
plt.figure(figsize=(8, 6), dpi=80)           
for i in range(0,3):    
    c = next(color)
    plt.scatter(compsensivity[i],negativities[i],c=c,cmap = 'viridis_r', label = (i+3))
    plt.ylabel("Negativity difference")
    plt.xlabel("Compass Sensitivity")
plt.legend()

plt.figure(figsize=(8, 6), dpi=80)
for i in range(3,16):    
    c = next(color)
    plt.scatter(compsensivity[i],negativities[i],c=c,cmap = 'viridis_r', label = (i+3))
    plt.ylabel("Negativity difference")
    plt.xlabel("Compass Sensitivity")
    plt.legend(loc='lower right')



# In[157]:


#finding correlation coefficents for negativity difference and mean

negcoeff = np.corrcoef(compsens, negat)
print(negcoeff)

meanegcoeff = np.corrcoef(compsens, negatmean)
print(meanegcoeff)


# In[160]:


#investigate the curling back which occurs at large values of compass sensitivity
#only works for original dataset with r = 17.86
interesting = []
position = []
interestingpoints = []

for i in range(len(compsens)):
    if compsens[i] > 0.14:
        interesting.append(i)
#saves the index of all rows which have a compass senstivity greater than 0.14      
    
result = np.zeros((len(interesting),3),dtype=float)


for j in interesting:
    interestingpoints.append(r[j])

pointss = np.asarray(interestingpoints)
#an array containing only the "interesting points"

    
plt.figure(figsize=(8, 6), dpi=80)
ax = plt.axes(projection = '3d')
ax.set(xlabel = 'x')
ax.set(ylabel= 'y')
ax.set(zlabel= 'z')
ax.scatter3D(result[:,0],result[:,1],result[:,2])


# In[161]:


#plotting compass sensitivity vs coherence difference for the interesting points

for i in interesting:
    diffcohwon.append(abs(coherencemaxwo[i] - coherenceminwo[i]))
newcompsens = []
for j in interesting:
    newcompsens.append(compsens[j])
newdiffcoh = []
for j in interesting:
    newdiffcoh.append(diffcoh[j])
newdiffcohwo = []
for j in interesting:
    newdiffcohwo.append(diffcohwo[j])
    
#plt.scatter(newcompsens,newdiffcoh)
plt.figure(1)
#plt.scatter(newcompsens,newdiffcohwo, c= 'r')
plt.title("Coherence difference vs compass sensitivity for large values of compass sensitivity only (without nuclear spin)")
plt.ylabel("Coherence difference")
plt.xlabel("Compass Sensitivity")
plt.figure(3)
#plt.scatter(newcompsens,newdiffcoh)
plt.title("Coherence difference vs compass sensitivity for large values of compass sensitivity only (with nuclear spin)")
plt.ylabel("Coherence difference")
plt.xlabel("Compass Sensitivity")


# In[15]:


#same as above for coherence mean
newcompsensm = []
for j in interesting:
    newcompsensm.append(compsens[j])
newdiffcohm = []
for j in interesting:
    newdiffcohm.append(meancoh[j])
newdiffcohwom = []
for j in interesting:
    newdiffcohwom.append(meancohwo[j])
    
#plt.scatter(newcompsens,newdiffcoh)
plt.figure(1)
plt.scatter(newcompsensm,newdiffcohwom, c= 'r')
plt.title("Coherence mean vs compass sensitivity for large values of compass sensitivity only (without nuclear spin)")
plt.ylabel("Coherence mean")
plt.xlabel("Compass Sensitivity")
plt.figure(3)
plt.scatter(newcompsensm,newdiffcohm)
plt.title("Coherence mean vs compass sensitivity for large values of compass sensitivity only (with nuclear spin)")
plt.ylabel("Coherence mean")
plt.xlabel("Compass Sensitivity")


# In[20]:


#plotting all the radical pair orientations in 3d 
import math
rabs2 = 17.86371856186401


def sphere(samples):

    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append((x, y, z))

    return np.array(points) *(rabs2)

data = sphere(300)


plt.figure(figsize=(10, 8), dpi=80)
ax = plt.axes(projection = '3d')
ax.set(xlabel = 'x')
ax.set(ylabel= 'y')
ax.set(zlabel= 'z')
                 

ax.scatter3D(data[:,0],data[:,1],data[:,2], cmap='viridis',s = 8, label = 'all points')
ax.scatter3D(result[:,0],result[:,1],result[:,2], c = 'r', s = 8, label = 'special points')
plt.legend()


# In[42]:


yz = []
def spherez(samples):
    spz = (17.843348900921036)**2 - 49
    z = 7.0
    xyz1 = []
    r = spz 

    theta = np.linspace(0,np.pi * 2,samples)
    for theta1 in theta:
    
        x = r * np.cos(theta1)
        y = r * np.sin(theta1)
        
        xyz1.append([x,y,z])
    yz = np.asarray(xyz1)
    ax = plt.axes(projection = '3d')
    ax.set(xlabel = 'x')
    ax.set(ylabel= 'y')
    ax.set(zlabel= 'z')
    ax.scatter3D(yz[:,0],yz[:,1],yz[:,2], s = 1)
    #print(yz)
    return np.array(yz)
 
spherez(1000)


# In[ ]:


#this code plots a 3d colour mapped plot of all radical pair orientations, maps coherence (can be chnaged to compass sensitivity)

mphys_df = pd.read_csv('mphys.csv', names = header)
#reads the text file

data = sphere(300)

c = diffcoh
#indicating that difference coherence will be used as color map

plt.figure(figsize=(10, 8), dpi=80)
ax = plt.axes(projection = '3d')
ax.set(xlabel = 'x')
ax.set(ylabel= 'y')
ax.set(zlabel= 'z')
p = ax.scatter3D(data[:,0],data[:,1],data[:,2], c=c, cmap = 'viridis_r',s = 20, label = 'all points')
#plot all the points on the sphere, using the coherence difference values to set the colour of each point
ax.scatter3D(result[:,0],result[:,1],result[:,2],c='r' ,s = 28, label = 'special points')
#plotting the special points on the same sphere
plt.legend()
fig.colorbar(p, ax=ax, label = 'Mean coherence')
ax.view_init(90, 90)
        
        
plt.figure(figsize=(10, 8), dpi=80)
ax = plt.axes(projection = '3d')
ax.set(xlabel = 'x')
ax.set(ylabel= 'y')
ax.set(zlabel= 'z')
p = ax.scatter3D(data[:,0],data[:,1],data[:,2], c=c, cmap = 'viridis_r',s = 20, label = 'all points')
ax.scatter3D(result[:,0],result[:,1],result[:,2],c='r' ,s = 28, label = 'special points')
plt.legend()
fig.colorbar(p, ax=ax, label = 'Mean coherence')
ax.view_init(-140, 170)
        
        
plt.figure(figsize=(10, 8), dpi=80)
ax = plt.axes(projection = '3d')
ax.set(xlabel = 'x')
ax.set(ylabel= 'y')
ax.set(zlabel= 'z')
p = ax.scatter3D(data[:,0],data[:,1],data[:,2], c=c, cmap = 'viridis_r',s = 20, label = 'all points')
ax.scatter3D(result[:,0],result[:,1],result[:,2],c='r' ,s = 28, label = 'special points')
plt.legend()
fig.colorbar(p, ax=ax, label = 'Mean coherence')
ax.view_init(-140, 80)
        
        
plt.figure(figsize=(10, 8), dpi=80)
ax = plt.axes(projection = '3d')
ax.set(xlabel = 'x')
ax.set(ylabel= 'y')
ax.set(zlabel= 'z')
p = ax.scatter3D(data[:,0],data[:,1],data[:,2], c=c, cmap = 'viridis_r',s = 20, label = 'all points')
ax.scatter3D(result[:,0],result[:,1],result[:,2],c='r' ,s = 28, label = 'special points')
plt.legend()
fig.colorbar(p, ax=ax, label = 'Mean coherence')
ax.view_init(180, 180)

        


# In[ ]:


#this code plots the maximum compass sensitivity for each magnitude and the minimum compass sensitivity for each mangitude agaisnt the magntidue
compsensmin = []
compsensmax = []
np.asarray(compsensivity[2])
are = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
print(len(are))
for j in range(0,16):
    compsensmin.append(min(compsensivity[j]))
    compsensmax.append(max(compsensivity[j]))
print(len(compsensmin)
      
plt.stackplot(are,compsensmin, compsensmax, labels=['Minimum compass sensitivity','Maximum compass sensitivity'])
plt.legend(loc='upper left')
plt.ylabel("Compass Sensitivity")
plt.xlabel("Magnitude of r")
plt.xlim(3,18)


# In[ ]:


#Lukes paper on coherence measures

