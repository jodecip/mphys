import qutip as qt
import numpy as np
import scipy as sp
import scipy.integrate as integrate
from matplotlib import pyplot as plt
import csv
import scipy.optimize as sp
from joblib import Parallel, delayed


# In[8]:


constants = dict(I = 1, g = 2.01325, muB = 9.274e-24, hbar = 1.0545718e-34)
#print(constants)


# In[2]:


def point_dipole_dipole_coupling(r):
    '''This function to Find Dipole-Dipole Coupling for a given distance r.
            Parameters:
                    r(np.array): 1d array [x,y,z] components of the vector r between the two radicals
                
            Returns:
                    A(np.array): 2d 3x3 matrix which is the electron-electron dipolar coupling tensor
    '''

    dr3 = -4*np.pi*1e-7 * (2.0023193043617 * 9.27400968e-24)**2 / (4*np.pi*1e-30)/6.62606957e-34/1e6 # MHz * A^3

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


# In[3]:


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


# In[4]:


def l1_norm_coherence(rho): 
    return np.sum(abs(rho-np.diag(np.diag(rho)))) # sum of absolute values of off-diagnoal matrix elements of density operator 

def traceNuclei(rho):
    if type(rho) is qt.Qobj:
        return qt.ptrace(rho, (0,1))
    else:
        n = 4 # 4 states for 2 electrons 
        m = rho.shape[0]//n
        return np.trace(rho.reshape(n,m,n,m), axis1=1, axis2=3) # removes nuclear-part from density operator

sqrt2 = np.sqrt(2)
basisSTee = np.array([[0, 1/sqrt2, -1/sqrt2, 0],
                      [1, 0,        0,       0],
                      [0, 1/sqrt2,  1/sqrt2, 0],
                      [0, 0,        0,       1]]).T # each row represents a state (S, T0, T+,T-)
basisST = np.kron(basisSTee, np.eye(3)) #this represents nuclear and electron spins
# used to transform up/down basis to singlet/triplet basis 

def calc_l1_norm_coherence(states):
        '''This function to calculate the coherence of a state.
            Parameters:
                    states(np.array): density matrix describing the state
                
            Returns:
                    c(np.array): array containing coherences for all the states
    '''
    c = []
    for state in states:
        tr = state.tr()
        #trace of the state
        rho_norm = state.data / tr
        #dividing by the trace to normalise
        rhoST = basisST.T @ (rho_norm @ basisST)
        #transforms the state into the singlet - triplet basis
        rhoSTee = traceNuclei(rhoST)
        #removes nuclear-part from the density matrix
        c.append((l1_norm_coherence(rhoST) * tr, l1_norm_coherence(rhoSTee) * tr))
        #caulcates coherence
        
    return c
    


# In[ ]:


def calc_negativity(states):
    '''This function to calculate the negativity (entanglement measuere) of a state.
            Parameters:
                    states(np.array): density matrix describing the state
                
            Returns:
                    N(np.array): array containing negativities for all the states
    '''
    N = []
    for state in states:
        tr = state.tr()
        rho_norm  = state / tr
        rho_norm_ee = traceNuclei(rho_norm)
        N.append(qt.negativity(rho, 1) * tr)
    return N


# In[5]:


def sy_diff_oris(r,j,kt): 
    '''This is the master function. For different values of r (vector pointing between the radicals), it finds the magentic field orientations which give the max and min singlet yields. For these orienatations of the magnetic field, coherence and entanglement measures are calculated
            Parameters:
                    r(np.array): 1d array [x,y,z] components of the vector r between the two radicals
                    j(int): magnitude of r
                    kt(int): value of kt (triplet decay rate constant)
                
            Returns:
                    filedata(np.array): array containing all the neccessary information about each different r vector.
    '''
    mag = j
    I = 1
    g = 2.01325
    muB = 9.274e-24 # J/T
    hbar = 1.0545718e-34  # m^2 kg / s
    efreq = 1e-9 * g*muB/hbar
    omega0 =  efreq * 50e-3 
    #omega0 = 1.4 * 2*np.pi # 1.4 MHz -> rad/us
    A = np.array([[-0.0995, 0.0029, 0],[0.0029, -0.0875, 0],[0, 0, 1.7569]]) * efreq # Mrad/s
    ks = 1 # singlet state decay constant /us
    #kt = 100 # triplet state decay constant /us
    # kt >> ks otherwise unchanged from before - look for a good value for kt
    Jxyz = (qt.spin_Jx, qt.spin_Jy, qt.spin_Jz)
    S1xyz = [qt.tensor(op(0.5), qt.identity(2), qt.identity(2*I+1)) for op in Jxyz]
    S2xyz = [qt.tensor(qt.identity(2), op(0.5), qt.identity(2*I+1)) for op in Jxyz]
    S12xyz = [S1xyz[i] + S2xyz[i] for i in range(3)]
    up = qt.basis(2,0)
    down = qt.basis(2,1)
    singlet = (qt.tensor(up,down) - qt.tensor(down,up))/np.sqrt(2)
    Ps12 = singlet * singlet.dag() # |S><S|
    Ps = qt.tensor(Ps12, qt.identity(2*I+1))
    Pt = 1 - Ps #Triplet projection operator, with 1 as the identity operator
    rho0 = Ps / Ps.tr() # 1/3 |S,nuc1><S,nuc1| + 1/3 |S,nuc0><S,nuc0| + 1/3 |S,nuc-1><S,nuc-1| 
    tlist = np.linspace(0, 14/ks, 14001)
    yields = []
    coherences = []
    negativities = []
    orismm =[]
    osimm = []
    
    Hhfc = 0
    for i in range(3):
        for j in range(3):
            if A[i,j] != 0:
                Hhfc += A[i,j] * qt.tensor(Jxyz[i](0.5), qt.identity(2), Jxyz[j](I))
                #hyperfine coupling hamiltonian 
    
    D = point_dipole_dipole_coupling(r) * 2*np.pi
     
    Heed = 0
    for i in range(3):
        for j in range(3):
            if D[i,j] != 0:
                Heed += D[i,j] * qt.tensor(Jxyz[i](0.5), Jxyz[j](0.5), qt.identity(3))
                #electron electron dipolar coupling hamiltonian 
                
    opt = qt.Options() 
    opt.store_states = True #function will store all states as time passes

    tlist = np.linspace(0, 14/ks, 14001) 
    
    oris = sphere(3,1)
    #generates magnetic field orientations
    
    K = (ks/2 * Ps) + (kt/2 * Pt)
    b = -np.asarray(qt.operator_to_vector(rho0).data.todense()).reshape(-1)
    yields = []
    for ori in oris:
        omega0vec = ori * omega0
        Hzee = sum(omega0vec[i] * S12xyz[i] for i in range(3))
        H = Hzee + Hhfc + Heed
        Heff = H - (1j * K)
        #effective Hamiltonian
        Leff = -1j*qt.spre(Heff) + 1j*qt.spost(Heff.conj().trans())
        A1 = Leff.data
        x = scipy.sparse.linalg.linsolve.spsolve(A1, b)
        x = qt.vec2mat(x)
        ys = ks * np.real( (Ps * qt.Qobj(x, dims=rho0.dims)).tr() )
        yields.append(ys)
        
    
    max2 = yields.index(max(yields))
    min2 = yields.index(min(yields))
    max3 = oris[yields.index(max(yields))]
    min3 = oris[yields.index(min(yields))]
    #this is the magnetic field orientation which gives the max and min singlet state yield
    orismm.append(max3)
    orismm.append(min3)
    

    coherences = []
    negativities = []
    states = []
    for ori in orismm:
        omega0vec = ori * omega0
        Hzee = sum(omega0vec[i] * S12xyz[i] for i in range(3))
        H = Hzee + Hhfc + Heed
        Heff = H - (1j * K)
        #effective Hamiltonian
        Heff_conj = Heff.conj().trans() 
        Leff = -1j * ( qt.superoperator.spre(Heff)- qt.superoperator.spost(Heff_conj))
        res = qt.mesolve(Leff, rho0, tlist, e_ops=[Ps], options=opt) 
        # singlet probability as function of time 
        ps = res.expect[0]
        ys = ks * integrate.simps(ps, tlist)
        c1t = calc_l1_norm_coherence(res.states)
        c1measures = integrate.simps(np.array(c1t), tlist, axis=0) 
        # yield of coherence measures
        coherences.append(c1measures)
        N = calc_negativity(res.states)
        Nmeasure = integrate.simps(np.array(N), tlist)
        negativities.append(Nmeasure)
    

    
    filedata = [mag, r, max(yields), max3, coherences[0][0], coherences[0][1], min(yields), min3, coherences[1][0], coherences[1][1], negativities]

    return filedata


# In[6]:


def write(j,number):
    '''This function runs the above sy_diff_oris function for all ropts for each relposition and writes the data to a csv file
            Parameters:
                    j(int): magnitude of r
                    number(int): the quantity of different r vectors wanted
    '''
    ropts = sphere(number,j)
    for r in ropts: 
        with open('difforisneg.csv'.format(number),'a') as f:
            writer = csv.writer(f)
            result = sy_diff_oris(r,j,1)
            writer.writerow(result)
            result = [[],[],[],[],[],[],[],[],[],[],[]]



#runs through different r values with parallelization
from joblib import Parallel, delayed
def sy_spherejob(number):
    '''Thius function runs the above sy_diff_oris function for all ropts for each relposition and writes the data to a csv file
            Parameters:
                    number(int): the quantity of different r vectors wanted
    '''
    
    

    header = ['Distance between Radicals','Orientation of Radicals', 'Maximum Singlet Yield', 'Orientation Responsible','Coherence with nuclear spin','Coherence without nuclear spin','Minimum Singlet Yield', 'Orientation Responsible','Coherence with nuclear spin','Coherence without nuclear spin']
    with open('difforisneg.csv'.format(number),'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
         
    relposition = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18] 
    #the distance between the radicals (in angstrom)
    result = Parallel(n_jobs=-1)(delayed(write)(j,number) for j in relposition)
      
            

sy_spherejob(50)



