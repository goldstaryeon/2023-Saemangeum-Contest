#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


river_data = pd.read_csv('새만금하천수질지표.csv', encoding='cp949')


# In[3]:


def cal_river_pH(q):
    if q > 6.5 and q <= 8.5:
        pollution = 1
    elif q > 6.0 and q <= 6.5:
        pollution = 5
    else:
        pollution = False
    return pollution


# In[4]:


def cal_river_BOD(q):
    if q <= 1:
        pollution = 1
    elif q <= 2:
        pollution = 2
    elif q <= 3:
        pollution = 3
    elif q <= 5:
        pollution = 4
    elif q <= 8:
        pollution = 5
    elif q <= 10:
        pollution = 6
    elif q > 10:
        pollution = 7
    return pollution


# In[5]:


def cal_river_SS(q):
    if q <= 25:
        pollution = 1
    elif q <= 100:
        pollution = 5
    else:
        pollution = False
    return pollution


# In[6]:


def cal_river_DO(q):
    if q >= 7.5:
        pollution = 1
    elif q >= 5.0:
        pollution = 2
    elif q >= 2.0:
        pollution = 5
    elif q < 2.0:
        pollution = 7
    return pollution


# In[7]:


def cal_river_TP(q):
    if q <= 0.02:
        pollution = 1
    elif q <= 0.04:
        pollution = 2
    elif q <= 0.1:
        pollution = 3
    elif q <= 0.2:
        pollution = 4
    elif q <= 0.3:
        pollution = 5
    elif q <= 0.5:
        pollution = 6
    elif q > 0.5:
        pollution = 7
    return pollution


# In[8]:


def cal_river_CG(q): #총대장균군
    if q <= 50:
        pollution = 1
    elif q <= 500:
        pollution = 2
    elif q <= 1000:
        pollution = 3
    elif q <= 5000:
        pollution = 4
    else:
        pollution = False
    return pollution


# In[9]:


def cal_river_FC(q): #분원성대장균군
    if q <= 10:
        pollution = 1
    elif q <= 100:
        pollution = 2
    elif q <= 200:
        pollution = 3
    elif q <= 1000:
        pollution = 4
    else:
        pollution = False
    return pollution


# In[10]:


def cal_mean_rate(row):
    rates = [rate for rate in row if rate is not False]
    return sum(rates) / len(rates)


# In[11]:


river_pol_df = pd.DataFrame({
    'BOD_pol': river_data["BOD(㎎/L)"].apply(cal_river_BOD),
    'SS_pol': river_data["SS(㎎/L)"].apply(cal_river_SS),
    'TP_pol': river_data["TP(㎎/L)"].apply(cal_river_TP),
    'CG_pol': river_data["총대장균군수(총대장균군수/100ml)"].apply(cal_river_CG),
    'FC_pol': river_data["분원성대장균군수"].apply(cal_river_FC)
})


# In[12]:


river_pol_df['mean_rate'] = round(river_pol_df.apply(cal_mean_rate, axis=1), 2)


# In[13]:


river_data['오염도'] = river_pol_df['mean_rate']


# In[14]:


river_data.sort_values(by='오염도', ascending=False, inplace=True)


# In[15]:


del river_data['DO(㎎/L)']
del river_data['수소이온농도']


# In[16]:


river_data.to_csv('새만금_하천_오염도.csv', index=False)


# In[20]:


get_ipython().run_line_magic('matplotlib', 'inline')
river_data


# In[18]:


import koreanize_matplotlib
get_ipython().run_line_magic('config', "inlineBackend.figure_format = 'retina'")


# In[19]:


x = list(river_data["측정소명"])
y = list(river_data["오염도"])

plt.figure(figsize = (15, 6))
plt.bar(x, y, color='#6495ED')

plt.title('새만금 하천별 오염도', fontsize=16, pad=20)
plt.xticks(rotation=45)

plt.xlabel('하천', labelpad=15)
plt.ylabel('오염도', labelpad=30, rotation=0)

plt.grid(False)
plt.savefig('pollution_bar.png', dpi=300, bbox_inches='tight')
plt.show()


# In[ ]:




