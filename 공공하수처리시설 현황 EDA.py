#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np


# In[42]:


data = pd.read_csv('한국환경공단_공공하수처리시설 현황_20211231.csv', encoding='ansi')


# In[43]:


data = data[data["세부단위구역"] == "새만금"]


# In[44]:


data = data[['시설명', '시설용량', '유입하수량', '방류량', '처리효율',
      '처리부하량', '처리방법', '사업비(백만원)', '지류',
     '연간 총 에너지 사용량_총 전력사용량(kWh)', '하수처리량당 이산화탄소(CO2)배출량']]


# In[45]:


data


# In[46]:


data['처리방법'].unique().tolist()


# In[47]:


data[data['처리방법'] == 'JASSFR']


# In[63]:


data['연간 총 에너지 사용량_총 전력사용량(kWh)'].mean()


# In[51]:


data['하수처리량당 이산화탄소(CO2)배출량'].mean()


# In[52]:


data['사업비(백만원)'].sum()


# In[59]:


data['처리효율'].mean()


# In[75]:


data[data['지류'] == '동진강']['연간 총 에너지 사용량_총 전력사용량(kWh)'].mean()


# In[74]:


data[data['지류'] == '원평천']['연간 총 에너지 사용량_총 전력사용량(kWh)'].mean()


# In[67]:


data[data['지류'] == '용암천']['연간 총 에너지 사용량_총 전력사용량(kWh)'].mean()


# In[68]:


data[data['지류'] == '용호천']['연간 총 에너지 사용량_총 전력사용량(kWh)'].mean()


# In[69]:


data[data['지류'] == '고부천']['연간 총 에너지 사용량_총 전력사용량(kWh)'].mean()


# In[70]:


data[data['지류'] == '마산천']['연간 총 에너지 사용량_총 전력사용량(kWh)'].mean()


# In[71]:


data[data['지류'] == '원평천']


# In[ ]:




