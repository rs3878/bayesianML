
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
get_ipython().run_line_magic('matplotlib', 'inline')


# So what we do here is the following:
# 
# We are given a matrix $R \in R^{m*n}$ such that $R_{ij} \in \{0,1\}$, and we want to decompose $R=UV$, where $U \in R^{m*d}$ and $V \in R^{d*n}$. 
# 
# The model and prior are the following:
# $$ R_{ij} \sim Ber(\Phi(u_i^*v_j/\sigma))$$
# $$ U_{i,:} \sim N(0,\lambda^{-1} I_m)$$
# $$ V_{:,j} \sim N(0,\lambda^{-1} I_n)$$

# Note there's a slight modification of the algorithm here. Now we have $(i,j) \in \{\text{given pairs}\}=\Omega$. So they are of the same dimension now. Let's call number of given pairs $n_{obv}$. So we have the $\phi_{(i,j)\in\Omega}$, a four way tensor, the $i$ dimension accounts for customer, the $j$ dimension accounts for, the $d$ dimension accounts for number of latent factors, and $b$ simply means if $y_{ij}=1$ or $0$. 
# 
# 
# basically the algorithm goes like the following.
# 
# 
# initialize
# 
# repeat the following until convergence:
# 
# 1) the E step
#     Do expectation $\forall \phi_{(i,j)\in\Omega}$.
#     
#     
# 2) the M step 
#     update U holding V constant
#     update V holding U constant
#     
#     
# 3) compute log likelihood (we gonna ignore this for now)

# In[2]:


from numpy import genfromtxt
my_data = genfromtxt('ratings.csv', delimiter=',')
import time
m = int(max(my_data[:,0]))
n = 1683
d = 5
n_obs = len(my_data)
phi = np.zeros((n_obs))
U = st.norm.rvs(scale = 0.1**0.5,size = (m,d))
V = st.norm.rvs(scale = 0.1**0.5,size = (d,n))


# In[3]:


test_data = genfromtxt('ratings_test.csv',delimiter=',')
n_test = test_data.shape[0]
data_test = np.zeros_like(test_data)
for i in range(n_test):
    if test_data[i,-1] == 1:
        data_test[i,-1] = 1
        
data_test[:,:2] = test_data[:,:2]-1
data_test = data_test.astype("int")


# In[4]:


data_01 = np.zeros_like(my_data)
for i in range(n_obs):
    if my_data[i,-1] == 1:
        data_01[i,-1] = 1
        
data_01[:,:2] = my_data[:,:2]-1
data_01 = data_01.astype("int")


# In[5]:


def E_step_vec(U,V,sigma,data,phi_stored):
    """
    VECTORIZED VERSION 
    
    we need:
        only a given set of data, u,v,sigma,
        U,V,d and sigma of course
    output:
        None, but we alter the phi_stored which stores
        E_qt phi, of shape 2*n_obs*d 
    
    """
    
    n_obs = data.shape[0]
    i_set, j_set = data[:,0],data[:,1]
    #UV = np.dot(U[i_set,:],V[:,j_set])
    UV = np.sum(U[i_set,:]*(V[:,j_set].T),axis=1)
    new_UV = -UV/(sigma)
    #one_vec = data[:,2] == 0
    pn_vec = np.ones(n_obs)
    pn_vec[data[:,2]==0] = -1
    
    wtv = sigma*pn_vec*np.exp(st.norm.logpdf(new_UV)-st.norm.logcdf(-new_UV*pn_vec))
    
    phi_stored = UV + wtv    
    
    return(phi_stored)


# In[6]:


def M_step(U,V,sigma,lamda,data,phi_stored):
    """
    Here we update U and V given the data
    """
    
    m,d = U.shape
    _,n = V.shape
    for i in range(m):
        # here we update U[i,:]
        relevant_ind = data[:,0]==i
        relevant_V = V[:,data[relevant_ind,1]]
        relevant_phi = phi_stored[relevant_ind]
        U_pt1 = lamda*np.eye(d) + relevant_V@relevant_V.T/(sigma**2)
        U_pt2 = (relevant_V@relevant_phi)/(sigma**2)
        new_U_i = np.linalg.inv(U_pt1)@U_pt2
        U[i,:] = new_U_i
    
    for j in range(n):
        relevant_ind = data[:,1]==j
        relevant_U = U[data[relevant_ind,0],:]
        relevant_phi = phi_stored[relevant_ind]
        V_pt1 = lamda*np.eye(d) + (relevant_U.T)@relevant_U/(sigma**2)
        V_pt2 = (relevant_U.T@relevant_phi)/(sigma**2)
        new_V_j = np.linalg.inv(V_pt1)@V_pt2
        V[:,j] = new_V_j
    
    return(U,V)     


# In[7]:


'''
This is likelihood of the parameters we want to maximize

def lower_bound(U,V,data,phi,lamda,sigma):
    """
    In theory, if the correct 
    """
    
    n_obs = data.shape[0]
    i_set, j_set = data[:,0],data[:,1]
    
    U_obs = U[i_set,:]
    V_obs = V[:,j_set].T
    pt1 = -lamda/2*(np.linalg.norm(U_obs,ord='fro')+np.linalg.norm(V_obs,ord='fro'))
    UV = np.sum(U_obs*V_obs,axis=1)
    pt2_1 = np.sum(UV**2)
    pt2_2 = 2*np.sum(UV*phi)
    pt2 = (pt2_2-pt2_1)/(2*sigma**2)
    
    return(pt1+pt2)     
'''


# In[13]:


def lower_bound(U,V,data,phi,lamda,sigma, d=5):
    """
    
    """
    m,d = U.shape
    n_obs = data.shape[0]
    i_set, j_set = data[:,0],data[:,1]
    
    
    U_obs = U[list(set(i_set)),:]    #list of set of i_set, so that we take UNIQUE is
    V_obs = V[:,list(set(j_set))].T  # same as above - UNIQUE js only
    
    pt1 = -1/lamda/2*(np.linalg.norm(U_obs,ord='fro')**2+np.linalg.norm(V_obs,ord='fro')**2)
    pt1 += (n+m)*d*np.log(1/lamda/2/np.pi)
    # now the frobenius norm **2 is the same thing as U_obv^T U_obv, same for V_obv
   
    UV = np.sum(U[i_set,:]*(V[:,j_set]).T,axis=1)    # this time we take unique (i,j) PAIRS
    pn_vec = np.ones(n_obs)    # same vectorization trick for performance
    pn_vec[data[:,2]==0] = -1

    pt2 = np.sum(st.norm.logcdf(UV*pn_vec/sigma))
    
    return(pt2+pt1)


# In[14]:


def pred_percentage(U,V,data,sigma):
    R = U@V
    i_set,j_set = data[:,0],data[:,1]
    pre_ber = st.norm.cdf(R[i_set,j_set]/sigma)
    pred = np.zeros_like(pre_ber)
    pred[pre_ber>=0.5] = 1
    pred[pre_ber<0.5] =0
    percent = (np.sum(np.abs(pred==data[:,-1]))/len(i_set))
    print("prediction correct percentage "+str(percent))


# In[10]:


def EM_pmf(data,dims,low_rank,sigma,lamda,T=100,verbose=False,full=False):
    
    m,n = dims 
    d = low_rank
    n_obs = len(data)
    
    #initialization
    phi = np.zeros((n_obs))
    U = st.norm.rvs(scale = 0.1,size = (m,d))
    V = st.norm.rvs(scale = 0.1,size = (d,n)) 
    
    if full:
        log_p_list = np.zeros(int(2*T))
    
    if verbose:
        import time
    
    for t in range(0,2*T,2):
        if verbose:
            start = time.time()
            print("===== iteration "+str(int(t/2)+1) +" =====")

        phi = E_step_vec(U,V,sigma,data_01,phi)
        
        if full: 
            log_p = lower_bound(U,V,data_01,phi,lamda,sigma)
            log_p_list[t] = log_p
        
        if verbose:
            mid = time.time()
            print("-----------------------------------------")
        
            if t != 0:
                print("|diff in log_p after E is " + str(log_p_list[t]-log_p_list[t-1])+" |")
            else:
                print("|diff in log_p after E is 0 |")
            print("-----------------------------------------")
        
        U,V = M_step(U,V,1,1,data_01,phi)
        if verbose:
            end = time.time()
        if full:
            log_p = lower_bound(U,V,data,phi,1,1)
            log_p_list[t+1] = log_p
        
        if verbose:
            new_end = time.time()
            print("the E step took "+str(mid-start)+"s")
            print("the $\phi$ looks like: \n"+str(phi))
            print("the M step took "+str(end-mid)+"s")
            print("the log posterior step took " +str(new_end-end)+"s")
            print("U looks like: \n"+str(U))
            print("V looks like: \n"+str(V))
            print("-----------------------------------------")
            print("|diff in log_p after M is " + str(log_p_list[t+1]-log_p_list[t])+" |")
            print("-----------------------------------------")
        
    if full:
        return(U,V,log_p,log_p_list)
    else:
        return(U,V)


# In[ ]:


U,V,log_lower_bound, log_p_list = EM_pmf(data_01,(m,n),5,1,1,verbose=False,full=True)


# Run your algorithm for 100 iterations and plot ln p(R, U, V ) for iterations 2 through 100.

# In[14]:


plt.plot([i for i in range(len(log_p_list[::2]))],log_p_list[::2])
plt.title("log likelihood converges?")
plt.xlabel("iteration #")
plt.ylabel("log likelihood")
plt.show()


# ### The results are the following

# In[15]:


pred_percentage(U,V,data_01,1)


# In[16]:


pred_percentage(U,V,data_test,1)


# Now let's do EM multiple times. Each 100 takes ~ $100*1.05s = 105s \approx 1.5 m $ to do. So 5 times will require about 7.5 minutes. 

# Rerun your algorithm for 100 iterations using 5 different random starting points. Plot the 5 different objective functions for iterations 20 through 100. Note: This is simply a repeat of Problem 2(a), only showing 5 objective functions instead of one and changing the x-axis.

# In[16]:


U,V,log_lower_bound, log_p_list1 = EM_pmf(data_01,(m,n),5,1,1,verbose=False,full=True)
U,V,log_lower_bound, log_p_list2 = EM_pmf(data_01,(m,n),5,1,1,verbose=False,full=True)
U,V,log_lower_bound, log_p_list3 = EM_pmf(data_01,(m,n),5,1,1,verbose=False,full=True)
U,V,log_lower_bound, log_p_list4 = EM_pmf(data_01,(m,n),5,1,1,verbose=False,full=True)
U,V,log_lower_bound, log_p_list5 = EM_pmf(data_01,(m,n),5,1,1,verbose=False,full=True)


# In[17]:


plt.plot([i for i in range(20,100)],log_p_list1[40::2], label ='1')
plt.plot([i for i in range(20,100)],log_p_list2[40::2], label ='2')
plt.plot([i for i in range(20,100)],log_p_list3[40::2], label ='3')
plt.plot([i for i in range(20,100)],log_p_list4[40::2], label ='4')
plt.plot([i for i in range(20,100)],log_p_list5[40::2], label ='5')
plt.title("log likelihood converges?")
plt.xlabel("iteration #")
plt.ylabel("log likelihood")
plt.legend(loc='best')
plt.show()


# Predict the values given in the test set and show your results in a confusion matrix. Show the raw counts in this confusion matrix.

# In[18]:


def pred(U,V,data,sigma):
    R = U@V
    i_set,j_set = data[:,0],data[:,1]
    pre_ber = st.norm.cdf(R[i_set,j_set]/sigma)
    pred = np.zeros_like(pre_ber)
    pred[pre_ber>=0.5] = 1
    pred[pre_ber<0.5] =0
    return pred


# In[19]:


from sklearn.metrics import confusion_matrix
pred = pred(U,V,data_test,1)
confusion_matrix(data_test[:,-1], pred)

