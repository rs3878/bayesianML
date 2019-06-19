
# coding: utf-8

# In[19]:


from scipy.stats import nbinom
import pandas as pd
import numpy as np
import scipy
import math
from decimal import *


# In[20]:


x_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("label_train.csv")
x_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("label_test.csv")


# In[21]:


x_train["y"] = y_train
x_test["y"] = y_test
y_train0 = x_train[x_train["y"] == 0]
y_train1 = x_train[x_train["y"]== 1]
y_test0 = x_test[x_test["y"] == 0]
y_test1 = x_test[x_test["y"] == 1]


# In[67]:


x_arr_train0 = np.array(y_train0)
n0 = x_arr_train0.shape[0]
x_arr_train1 = np.array(y_train1)
n1 = x_arr_train1.shape[0]
x_arr_test0 = np.array(y_test0)
x_arr_test1 = np.array(y_test1)
x_arr_test = np.array(x_test)


# In[64]:


def ystar_post(x_star, X0,X1, gamma_params, compute_prob = False):
    from scipy.stats import nbinom as neg_binom
    # we have y_star equal to 0 or 1, each have a probability
    # here we only care about the posterior of p(x^*|y^*=y, 
    # {x_i: y^*=y})
    
    a,b = gamma_params# gamma's parameters
    n0,d = X0.shape
    n1 = X1.shape[0]
    
    log_p_xstar_ystar_0 = np.sum(neg_binom.logpmf(
        x_star,a+np.sum(X0,axis=0),1-1/(1+b+n0)),axis=1)
    log_p_xstar_ystar_1 = np.sum(neg_binom.logpmf(
        x_star,a+np.sum(X1,axis=0),1-1/(1+b+n1)),axis=1)
    return(np.array((log_p_xstar_ystar_0,log_p_xstar_ystar_1)))


# In[23]:


def log_other_part(count_y0,count_y1,e,f):
    log_pt0 = np.log((f+count_y0))
    log_pt1 = np.log((e+count_y1))
    return(np.array((log_pt0,log_pt1)))


# In[35]:


# nbc for naive bayes classifier
def nbc(x_star, X0,X1, gamma_params,beta_params):
    e,f = beta_params
    n0,d = X0.shape
    n1 = X1.shape[0]
    # we compute log prob
    pt1 = ystar_post(x_star, X0,X1, gamma_params)
    pt2 = log_other_part(n0,n1,e,f)
    final = pt1+pt2.reshape((2,1))
    return(np.argmax(final,axis = 0))


# In[85]:


#accuracy
n,_ = x_arr_test.shape
gamma_params=(1,1)
beta_params= (1,1)
pred = (nbc(x_arr_test[:,:-1],x_arr_train0[:,:-1],x_arr_train1[:,:-1],
            gamma_params,beta_params))
sum(pred ==x_arr_test[:,-1])/460


# In[28]:


#confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(x_arr_test[:,-1], pred)


# In[29]:


#find three misclassified emails
wrong = []
for i in range(460):
    if pred[i] != x_arr_test[i,-1]:
        wrong.append(i)
print(wrong[:3])


# In[104]:


#predictive probability to check if it is =0
def find_pred_prob(x_star, X0,X1, gamma_params,beta_params,sam1,sam2,sam3):
    # nbc for naive bayes classifier
    e,f = beta_params
    n0,d = X0.shape
    n1 = X1.shape[0]
    # we compute log prob
    pt1 = ystar_post(x_star, X0,X1, gamma_params)
    pt2 = log_other_part(n0,n1,e,f)
    final = pt1+pt2.reshape((2,1))
    for i in [sam1,sam2,sam3]:
        print(math.exp(final[0][i])/(math.exp(final[0][i])+ math.exp(final[1][i])))
find_pred_prob(x_arr_test[:,:-1],x_arr_train0[:,:-1],
              x_arr_train1[:,:-1],gamma_params,beta_params,1,24,49)


# In[52]:


# 54 key words
with open('README') as f:
    file=f.read()

file = file.split('\n')
file = file[3:-1]
len(file)


# In[101]:


#plot three misclassified emails
import matplotlib.pyplot as plt
s1 = x_arr_test[1][:-1]
a = b = np.array(list(range(54)))
num0 = np.array(list(range(54)))
num1 = np.array(list(range(54)))
num0.fill(n0)
num1.fill(n1)
np.sum(x_arr_train0[:,:-1],axis=0)
plt.scatter(file, s1, alpha=0.5, label = "misclassified email")
plt.scatter(file, (np.sum(x_arr_train0[:,:-1], axis=0)+a)/(num0+b), alpha=0.5, label = "Exp lam0")
plt.scatter(file, (np.sum(x_arr_train1[:,:-1], axis=0)+a)/(num1+b), alpha=0.5, label = "Exp lam1")
plt.xticks(rotation = 40)
plt.legend()
fig = plt.gcf() 
fig.set_size_inches(20,15)


# In[102]:


s2=x_arr_test[24][:-1]
plt.scatter(file, s2, alpha=0.5, label = "misclassified email")
plt.scatter(file, (np.sum(x_arr_train0[:,:-1], axis=0)+a)/(num0+b), alpha=0.5, label = "Exp lam0")
plt.scatter(file, (np.sum(x_arr_train1[:,:-1], axis=0)+a)/(num1+b), alpha=0.5, label = "Exp lam1")
plt.xticks(rotation = 40)
plt.legend()
fig = plt.gcf() 
fig.set_size_inches(20,15)


# In[89]:


s3=x_arr_test[49][:-1]
plt.scatter(file, s3, alpha=0.5, label = "misclassified email")
plt.scatter(file, (np.sum(x_arr_train0[:,:-1], axis=0)+a)/(num0+b), alpha=0.5, label = "Exp lam0")
plt.scatter(file, (np.sum(x_arr_train1[:,:-1], axis=0)+a)/(num1+b), alpha=0.5, label = "Exp lam1")
plt.xticks(rotation = 40)
plt.legend()
fig = plt.gcf() 
fig.set_size_inches(20,15)


# In[90]:


#find the three most ambiguous emails
def find_max_diff(x_star, X0,X1, gamma_params,beta_params):
    # nbc for naive bayes classifier
    e,f = beta_params
    n0,d = X0.shape
    n1 = X1.shape[0]
    # we compute log prob
    pt1 = ystar_post(x_star, X0,X1, gamma_params)
    pt2 = log_other_part(n0,n1,e,f)
    final = pt1+pt2.reshape((2,1))
    diff = []
    for i in range(460):
        diff.append(abs((final[0][i]-final[1][i])))
    print(np.argsort(diff)[:3])
    
    
find_max_diff(x_arr_test[:,:-1],x_arr_train0[:,:-1],
              x_arr_train1[:,:-1],gamma_params,beta_params)


# In[105]:


#three predictive probability for the most ambiguous emails
find_pred_prob(x_arr_test[:,:-1],x_arr_train0[:,:-1],
              x_arr_train1[:,:-1],gamma_params,beta_params,391,430,396)


# In[81]:


s1=x_arr_test[391][:-1]
plt.scatter(file, s1, alpha=0.5, label = "misclassified email")
plt.scatter(file, (np.sum(x_arr_train0[:,:-1], axis=0)+a)/(num0+b), alpha=0.5, label = "Exp lam0")
plt.scatter(file, (np.sum(x_arr_train1[:,:-1], axis=0)+a)/(num1+b), alpha=0.5, label = "Exp lam1")
plt.xticks(rotation = 40)
plt.legend()
fig = plt.gcf() 
fig.set_size_inches(20,15)
plt


# In[80]:


s2=x_arr_test[430][:-1]
plt.scatter(file, s2, alpha=0.5, label = "misclassified email")
plt.scatter(file, (np.sum(x_arr_train0[:,:-1], axis=0)+a)/(num0+b), alpha=0.5, label = "Exp lam0")
plt.scatter(file, (np.sum(x_arr_train1[:,:-1], axis=0)+a)/(num1+b), alpha=0.5, label = "Exp lam1")
plt.xticks(rotation = 40)
plt.legend()
fig = plt.gcf() 
fig.set_size_inches(20,15)
plt


# In[79]:


s3=x_arr_test[396][:-1]
plt.scatter(file, s3, alpha=0.5, label = "misclassified email")
plt.scatter(file, (np.sum(x_arr_train0[:,:-1], axis=0)+a)/(num0+b), alpha=0.5, label = "Exp lam0")
plt.scatter(file, (np.sum(x_arr_train1[:,:-1], axis=0)+a)/(num1+b), alpha=0.5, label = "Exp lam1")
plt.xticks(rotation = 40)
plt.legend()
fig = plt.gcf() 
fig.set_size_inches(20,15)
plt

