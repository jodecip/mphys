#!/usr/bin/env python
# coding: utf-8

# In[2]:


import qutip as qt
import numpy as np
import scipy as sp
import scipy.integrate as integrate
from matplotlib import pyplot as plt
import csv
import time
from scipy import sparse




r = np.array([8.51,-14.25,6.55])

rabs2 = np.linalg.norm(r)
print(rabs2)



def l1_norm_coherence(rho):
    return np.sum(abs(rho-np.diag(np.diag(rho))))
#defining a sum of absolute values of off-diaglonal matrix elements of density operator

def traceNuclei(rho):
    if type(rho) is qt.Qobj:
        return qt.ptrace(rho, (0,1))
    else:
        n = 4
        m = rho.shape[0]//n
        return np.trace(rho.reshape(n,m,n,m), axis1=1, axis2=3)
#defining a func to remove nuclear spin, to see if it effects coherence measures (subsec)  



#this transforms basis from spin(up,down) to S-T basis

k0 = 1
ks =  1 #singlet state decay constant
kt =  1 #triplet state decay constant

def calc_l1_norm_coherence(states):
    
    c = []
    basisSTee = np.array([[0, 1/sqrt2, -1/sqrt2, 0],
                      [1, 0,        0,       0],
                      [0, 1/sqrt2,  1/sqrt2, 0],
                      [0, 0,        0,       1]]).T
    basisST = np.kron(basisSTee, np.eye(3))
    
def calc_l1_norm_coherence(states):
    c = []
    for state in states:
        rhoST = basisST.T @ (state.data @ basisST)
        rhoSTee = traceNuclei(rhoST)
        #this removes nuclear part
        c.append((l1_norm_coherence(rhoST), l1_norm_coherence(rhoSTee)))
    return c
#defining a function to calculate a coherence measure (l1 norm coherence)





def sphere(samples,rabs):
    '''This function finds random points on a sphere for radius 2 * abs).
            Parameters:
                    samples(int): The number of [x,y,z] vectors needed.
                    rabs(float): The magnitude of the desired vector.
                
            Returns:
                    points(np.array): 1d array [x,y,z]
    
    '''
    

    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append((x, y, z))

    return np.array(points) *  (rabs)



def point_dipole_dipole_coupling(r):
#assume spins are at well defined points in space, in reality they are delocalised. Using centre of spin density.

    dr3 = -4*np.pi*1e-7 * (2.0023193043617 * 9.27400968e-24)**2 / (4*np.pi*1e-30)/6.62606957e-34/1e6 # MHz * A^3
#dr3 contains all constants involved in the hamtitonian for electron-electron dipolar coupling
    
    #this if statement normalises A
    if np.isscalar(r):
    # assume r is aligned with z
        d = dr3 / r**3
        A = np.diag([-d, -d, 2*d])
    else:
        norm_r = np.linalg.norm(r)
        d = dr3 / norm_r**3
        e = r / norm_r
        A = d * (3 * e[:,np.newaxis] * e[np.newaxis,:] - np.eye(3))

    return A




def sy_sphere(number):
    
    
    header = ['ps','psrec','yields']
    #header = ['Orientation of Radicals', 'Maximum Singlet Yield', 'Orientation Responsible','Coherence with nuclear spin','Coherence without nuclear spin','Minimum Singlet Yield', 'Orientation Responsible','Coherence with nuclear spin','Coherence without nuclear spin']
    with open('syield.csv'.format(number),'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    ropts = sphere(number,rabs2)
    rmm = []

    for r in ropts:
        #rmm.append(sy_diff_oris(i))
        print(r)
        with open('syield.csv'.format(number),'a') as f:
            writer = csv.writer(f)
            writer.writerow(sy_diff_oris(r))
        



def sy_diff_oris_new(r,A,j):
    
    mag = j
   # Mrad/s
    ks = 1
    kt = 1
    I = 1
    g = 2.01325
    muB = 9.274e-24 # J/T
    hbar = 1.0545718e-34  
    efreq = 1e-9 * g*muB/hbar
    omega0 =  efreq * 50e-3 
    #larmor frequency
    A = np.array([[-0.0995, 0.0029, 0],[0.0029, -0.0875, 0],[0, 0, 1.7569]]) * efreq 
    Jxyz = (qt.spin_Jx, qt.spin_Jy, qt.spin_Jz)
    S1xyz = [qt.tensor(op(0.5), qt.identity(2), qt.identity(2*I+1)) for op in Jxyz]
    S2xyz = [qt.tensor(qt.identity(2), op(0.5), qt.identity(2*I+1)) for op in Jxyz]
    #components of spin
    S12xyz = [S1xyz[i] + S2xyz[i] for i in range(3)]

    up = qt.basis(2,0)
    down = qt.basis(2,1)
    #tensor(A,B) = A (x) B
    singlet = (qt.tensor(up,down) - qt.tensor(down,up))/np.sqrt(2)
    Ps12 = singlet * singlet.dag()
    Ps = qt.tensor(Ps12, qt.identity(2*I+1))
    #singlet projection operator
    Pt = 1 - Ps
    rho0 = Ps / Ps.tr() 
    #density operator of initial state
    sqrt2 = np.sqrt(2)

    Hhfc = 0
    for i in range(3):
        for j in range(3):
            if A[i,j] != 0:
                Hhfc += A[i,j] * qt.tensor(Jxyz[i](0.5), qt.identity(2), Jxyz[j](I))

    D = point_dipole_dipole_coupling(r) * 2*np.pi
    Heed = 0
    for i in range(3):
        for j in range(3):
            if D[i,j] != 0:
                Heed += D[i,j] * qt.tensor(Jxyz[i](0.5), Jxyz[j](0.5), qt.identity(3))
    opt = qt.Options()
    opt.store_states = True
    #stores all states
#ft is the sum of the states
    tlist = np.linspace(0, 14/ks, 14001)
    yields = []
    coherences = []
    negativities = []
        

  
    orismm =[]


    oris = sphere(3,1)
    one = qt.tensor(*(qt.identity(d) for d in [2,2,2*I+1]))
    K = k0/2 * one
  

    b = -np.asarray(qt.operator_to_vector(rho0).data.todense()).reshape(-1)


    for ori in oris:
        omega0vec = ori * omega0
        Hzee = sum(omega0vec[i] * S12xyz[i] for i in range(3))
        H = Hzee + Hhfc + Heed
        Heff = H - 1j * K
        Leff = -1j*qt.spre(Heff) + 1j*qt.spost(Heff.conj().trans())
        S = Leff.data
        x = sparse.linalg.linsolve.spsolve(S, b)
        x = qt.vec2mat(x)
        ys = k0 * np.real( (Ps * qt.Qobj(x, dims=rho0.dims)).tr() )
        yields.append(ys)
        
        
   
    max2 = yields.index(max(yields))
    min2 = yields.index(min(yields))
    max3 = oris[yields.index(max(yields))]
    min3 = oris[yields.index(min(yields))]
    #this is the magnetic field orientation which gives the max and min singlet state yield
    orismm.append(max3)
    orismm.append(min3)


    for ori in orismm:
        omega0vec = ori * omega0
        Hzee = sum(omega0vec[i] * S12xyz[i] for i in range(3))
        H = Hzee + Hhfc + Heed
        res = qt.mesolve(H, rho0, tlist, e_ops=[Ps], options=opt)
        ps = res.expect[0]
        psrec = ft * ps
        ys = integrate.simps(psrec, tlist)
        yields.append(ys)
        c1t = calc_l1_norm_coherence(res.states)
        c1measures = integrate.simps(ft[:,np.newaxis] * np.array(c1t), tlist, axis=0)

        
        
    filedata = [mag, r, max(yields), max3, coherences[0][0], coherences[0][1], min(yields), min3, coherences[1][0], coherences[1][1]]

    return filedata

A = np.array([[-0.0995, 0.0029, 0],[0.0029, -0.0875, 0],[0, 0, 1.7569]]) * efreq # Mrad/s




#this function runs the sy_diff_oris function for all ropts for each relposition and writes the data to a csv file
def write(j,number):
    ropts = sphere(number,j)
    for r in ropts: 
        with open('diffrandmksktfinal4.csv'.format(number),'a') as f:
            writer = csv.writer(f)
            result = sy_diff_oris_new(r,A,j)
            writer.writerow(result)
            result = [[],[],[],[],[],[],[],[],[],[]]



#runs through different r values with parallelization
from joblib import Parallel, delayed
def sy_spherejob(number):
    
    

    header = ['Distance between Radicals','Orientation of Radicals', 'Maximum Singlet Yield', 'Orientation Responsible','Coherence with nuclear spin','Coherence without nuclear spin','Minimum Singlet Yield', 'Orientation Responsible','Coherence with nuclear spin','Coherence without nuclear spin']
    with open('diffrandmksktfinal4.csv'.format(number),'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
         
    relposition = [3,4] 
    result = Parallel(n_jobs=1)(delayed(write)(j,number) for j in relposition)
      
            

sy_spherejob(3)


# In[ ]:




