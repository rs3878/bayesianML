
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scipy as sp
from scipy import special as spsp
from scipy import stats as spst
import matplotlib.pyplot as plt
from scipy.special import digamma, gammaln
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#hyperparameter 
a0 = b0 = 10**(-16)
e0 = f0 = 1

#load data
x1 = np.genfromtxt('X_set1.csv', delimiter=',')
x2 = np.genfromtxt('X_set2.csv', delimiter=',')
x3 = np.genfromtxt("X_set3.csv", delimiter=',')
y1 = np.genfromtxt("y_set1.csv", delimiter=',')
y2 = np.genfromtxt("y_set2.csv", delimiter=',')
y3 = np.genfromtxt("y_set3.csv", delimiter=',')
z1 = np.genfromtxt("z_set1.csv", delimiter=',')
z2 = np.genfromtxt("z_set2.csv", delimiter=',')
z3 = np.genfromtxt("z_set3.csv", delimiter=',')
x1 = np.array(x1)
x2 = np.array(x2)
x3 = np.array(x3)
y1 = np.array(y1)
y2 = np.array(y2)
y3 = np.array(y3)
z1 = np.array(z1)
z2 = np.array(z2)
z3 = np.array(z3)


# In[3]:


def read_data(X, Y):
    n,d = X.shape
    A0 = np.repeat(a0, d)
    B0 = np.repeat(b0, d)
    return n, d, A0, B0


# In[4]:


def q_lambda(e0, f0, n, Y, X, U, sigma):
    e = e0 + n/2
    f = f0 + 1/2*((Y-X@U).T@(Y-X@U) + np.trace(X@sigma@(X.T)))
    return e,f


# In[5]:


def q_alpha(A0, B0, sigma, mu):
    A = A0 + 1/2
    B = B0 + 1/2 * (np.diag(sigma)+ mu**2)
    return A,B


# In[6]:


def q_w(A, B, e, f, X, Y):
    sigma_inv = np.diag(A/B) + (e/f)*(X.T@X)
    sigma = np.linalg.inv(sigma_inv)
    mu = sigma@((e/f)*(X.T@Y))
    return mu,sigma


# In[16]:


def ln_p(A, B, e0, f0, e, f, a0, b0, mu, sigma, X, Y, n):
    p_w_1 = (1/2)*np.sum(spsp.digamma(A) - np.log(B))
    p_w =  p_w_1 -(1/2)*np.trace((sigma + np.outer(mu,mu.T))@np.diag(A/B))
    
    p_lambda = (e0-1)*(spsp.digamma(e) - np.log(f)) - f0*(e/f)
    
    p_alpha = (a0-1)*np.sum(spsp.digamma(A)-np.log(B))-np.sum(b0*(A/B))

    p_y_1 = (e/f) * np.trace(X@sigma@X.T + np.outer((X@mu-Y),((X@mu-Y).T)))
    p_y = (-1/2) * p_y_1 + (n/2)*(spsp.digamma(e) - np.log(f))
    
    result = p_w + p_lambda + p_alpha + p_y
    return result 


# In[17]:


def entropy(e, f, A, B, mu, sigma):
    entro_lambda = e - np.log(f) + gammaln(e) - (e - 1) * digamma(e) 

    entro_alpha = np.sum(A - np.log(B) + gammaln(A) - (A - 1) * digamma(A))
    
    sign, value = np.linalg.slogdet(sigma)
    entro_w = (1/2) * sign * value

    total_entropy = entro_lambda + entro_alpha + entro_w
    return total_entropy


# In[18]:


def vari_infer(A0, B0, U, n, sigma, X, Y, e, f, A, B, a0, b0, e0, f0, likelihood_list):
    e, f = q_lambda(e0, f0, n, Y, X, U, sigma)
    A, B = q_alpha(A0, B0, sigma, U)
    U, sigma = q_w(A, B, e, f, X, Y)

    obj_fcn_lnp = ln_p(A, B, e0, f0, e, f, a0, b0, U, sigma, X, Y, n)
    obj_fcn_entro = entropy(e, f, A, B, U, sigma)
    result = obj_fcn_lnp + obj_fcn_entro
    
    likelihood_list.append(result)
    return A,B,e,f,U,sigma


# In[19]:


def call(X, Y, T=500):
    #read dimension
    n, d, A0, B0 = read_data(X, Y)
    
    #initialization
    U = np.zeros(d)
    sigma = np.identity(d)
    A = B = np.ones(d)
    e = f = 1
    
    likelihood_list =[]
    
    for i in range(T):
        A,B,e,f,U,sigma = vari_infer(A0, B0, U, n, sigma, X, Y, e, f, A, B, a0, b0, e0, f0, likelihood_list)
    
    return likelihood_list, A, B, e, f, U, sigma, d, n, X, Y


# In[20]:


def plot_vi(likelihood_list):
    plt.plot(range(500), likelihood_list)
    plt.xlabel('Iterations')
    plt.title('Variational Objective Function')


# In[21]:


def plot_alpha(d,A,B):
    plt.plot(range(d), B/A)
    plt.xlabel('k')
    plt.ylabel('1/1/Eq[αk]')
    plt.title('1/Eq[αk] as a function of k.')


# In[22]:


likelihood_list_1, A, B, e, f, U, sigma, d, n, X, Y= call(x1,y1)
plot_vi(likelihood_list_1)


# In[23]:


plot_alpha(d,A,B)


# In[24]:


#1/Eq[λ] 
f/e


# In[25]:


y = X@U
plt.plot(z1, y, label = "X@U")
plt.plot(z1, Y, '.', label = "Y")
plt.plot(z1, 10*np.sinc(z1), label = "sinc(z)")
plt.legend()
plt.xlabel('z1')
plt.ylabel('y')
plt.title('Y vs. Z')


# In[26]:


likelihood_list_2, A, B, e, f, U, sigma, d, n, X, Y= call(x2, y2)
plot_vi(likelihood_list_2)


# In[27]:


plot_alpha(d,A,B)


# In[28]:


#1/Eq[λ] 
f/e


# In[29]:


y = X@U
plt.plot(z2, y, label = "X@U")
plt.plot(z2, Y, '.', label = "Y")
plt.plot(z2, 10*np.sinc(z2), label = "sinc(z)")
plt.legend()
plt.xlabel('z2')
plt.ylabel('y')
plt.title('Various Ys against Z')


# In[30]:


likelihood_list_3, A, B, e, f, U, sigma, d, n, X, Y= call(x3,y3)
plot_vi(likelihood_list_3)


# In[31]:


plot_alpha(d,A,B)


# In[33]:


#1/Eq[λ] 
f/e


# In[34]:


y = X@U
plt.plot(z3, y, label = "X@U")
plt.plot(z3, Y, '.', label = "Y")
plt.plot(z3, 10*np.sinc(z3), label = "sinc(z)")
plt.legend()
plt.xlabel('z3')
plt.ylabel('y')
plt.title('Various Ys against Z')

