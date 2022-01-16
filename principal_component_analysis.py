#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Principal Component Analysis #

# Steps:
    # 1) MxN matrix - M samples of N features
    # 2) Standardized mean-deviation matrix
    # 3) Covariance matrix
    # 4) Eigendecomposition of covariance matrix
    # 5) Dimension reduction


# In[2]:


# Step 1: MxN sample-feature matrix

import pandas as pd

df = pd.read_csv('weight-height.csv')
df.head()


# In[20]:


# Visualize height and weight

df.plot.scatter('Height', 'Weight', s=1)

# Visualize by gender
color_dict = {'Male': 'b', 'Female': 'r'}
df.plot.scatter('Height', 'Weight', s=1, c=[color_dict[x] for x in df['Gender']])


# In[24]:


# Generate sample-feature matrix

d = df[['Height', 'Weight']]


# In[35]:


# Step 2: Standardized mean-deviation matrix
    # for each column of d, calculate the mean and standard deviation
    # for each entry of each column of d, substract it from the mean and divide by the standard deviation
    
mean_height = d['Height'].mean()
mean_weight = d['Weight'].mean()
std_height = d['Height'].std()
std_weight = d['Weight'].std()

for row in range(len(d)):
    d.loc[row, 'Height'] = ((d.loc[row, 'Height'] - mean_height) / std_height)
    d.loc[row, 'Weight'] = ((d.loc[row, 'Weight'] - mean_weight) / std_height)
    
d


# In[37]:


d.loc[row, 'Height'].dtype


# In[ ]:





# In[40]:


# Visualize

d.plot.scatter('Height', 'Weight', s=1)


# In[52]:


# Step 3: Covariance Matrix
    # C = (transpose(N) * N) / n-1
    
C = pd.DataFrame(columns=['Height', 'Weight'], index=['Height', 'Weight'])
C.loc['Height', 'Height'] = ((d['Height'] * d['Height']).sum()) / (len(d) - 1)
C.loc['Height', 'Weight'] = ((d['Height'] * d['Weight']).sum()) / (len(d) - 1)
C.loc['Weight', 'Height'] = ((d['Weight'] * d['Height']).sum()) / (len(d) - 1)
C.loc['Weight', 'Weight'] = ((d['Weight'] * d['Weight']).sum()) / (len(d) - 1)
C


# In[67]:


# Step 4: Eigendecomposition of C

import numpy as np

A = C.to_numpy(dtype=np.float64)

eigenvalues, eigenvectors = np.linalg.eig(A)
print(eigenvalues)
print(eigenvectors)


# In[102]:


# Plot principal components

import matplotlib.pyplot as plt

x1 = [0, eigenvectors[0][0]]
y1 = [0, eigenvectors[0][1]]
x2 = [0, eigenvectors[1][0]]
y2 = [0, eigenvectors[1][1]]

print(x2, y2)
plt.plot(x1,y1)
plt.plot(x2,y2)


# In[103]:


# Plot w/ data

plt.scatter(d['Height'], d['Weight'], s=1)
plt.plot(x1,y1, LineWidth=1, c='black')
plt.plot(x2,y2, LineWidth=1, c='black')


# In[104]:


# Plot w/ data, but scale the vectors with their eigenvalue

nx1 = [i * eigenvalues[1] for i in x1]
ny1 = [i * eigenvalues[1] for i in y1]

nx2 = [i * eigenvalues[0] for i in x2]
ny2 = [i * eigenvalues[0] for i in y2]

plt.scatter(d['Height'], d['Weight'], s=1)
plt.plot(nx1,ny1, LineWidth=1, c='black')
plt.plot(nx2,ny2, LineWidth=1, c='black')


# In[106]:


# Variance accounted for

v = eigenvalues[1] / sum(eigenvalues)
v


# In[ ]:


# Step 5: Dimension Reduction
    # We see that 96% of the variance in height and weight can be expressed as a linear combination of the two

