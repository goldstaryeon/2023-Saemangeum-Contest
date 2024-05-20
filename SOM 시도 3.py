#!/usr/bin/env python
# coding: utf-8

# In[42]:


from minisom import MiniSom
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from matplotlib.patches import RegularPolygon
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colorbar as colorbar


# In[2]:


df_river = pd.read_csv('새만금_하천_오염도.csv')
df_river = df_river.iloc[:, 3:]


# In[3]:


df_river.iloc[6, 3] = 1000
df_river.iloc[7, 3] = 1000
df_river.iloc[6, 4] = 200
df_river.iloc[7, 4] = 200


# In[4]:


df_river.shape


# In[5]:


data = (df_river-np.mean(df_river, axis=0)) / np.std(df_river, axis=0)
data


# In[6]:


map_n = [n for n in range(2, 4)]
para_sigma = [np.round(sigma*0.1, 2) for sigma in range(1, 10)]
para_learning_rate = [np.round(learning_rate*0.1, 2) for learning_rate in range(1, 10)]


# In[7]:


res = []

for n in map_n:
    for sigma in para_sigma:
        for lr in para_learning_rate:
            try:
                estimator = MiniSom(n, n, 6, sigma=sigma, learning_rate = lr,
                                   topology='hexagonal', random_seed=0)
                estimator.random_weights_init(data.values)
                estimator.train(data.values, 1000, random_order=True)
                qe = estimator.quantization_error(data.values)
                winner_coordinates = np.array([estimator.winner(x) for x in data.values]).T
                cluster_index = np.ravel_multi_index(winner_coordinates, (n, n))
                res.append([str(n) + 'x' + str(n), sigma, lr, 'random_init', qe,
                           len(np.unique(cluster_index))])
                
            except ValueError as e:
                print(e)


# In[22]:


df_res = pd.DataFrame(res, columns=['map_size', 'sigma', 'learaning_rate',
                                   'init_method', 'qe', 'n_cluster'])
df_res.sort_values(by=['qe'], ascending=True, inplace=True, ignore_index=True)
df_res[df_res['n_cluster'] == 4]


# In[26]:


plt.figure(figsize=(20, 10))
sns.lineplot(data = df_res)


# In[27]:


som_b2 = MiniSom(2, 2, 6, sigma=0.9, learning_rate=0.5, topology='hexagonal',
                neighborhood_function='gaussian', activation_distance='euclidean',
                random_seed=0)
som_b2.random_weights_init(data.values)
som_b2.train(data.values, 1000, random_order=True)


# In[24]:


som_b2.quantization_error(data.values)


# In[47]:


xx, yy = som_b2.get_euclidean_coordinates()
umatrix = som_b2.distance_map()
weights = som_b2.get_weights()
winner_coordinates = np.array([som_b2.winner(x) for x in data.values]).T
cluster_index = np.ravel_multi_index(winner_coordinates, (2, 2))

f = plt.figure(figsize=(10, 10))
ax = f.add_subplot(111)
ax.set_aspect('equal')

for i in range(weights.shape[0]):
    for j in range(weights.shape[1]):
        wy = yy[(i, j)] * 2 / np.sqrt(3) * 3 / 4
        hex = RegularPolygon((xx[(i, j)], wy),
                            numVertices=6,
                            radius=.95 / np.sqrt(3),
                            facecolor=cm.Blues(umatrix[i, j]),
                            alpha=.4,
                            edgecolor='gray')
        plot = ax.add_patch(hex)
        

for c in np.unique(cluster_index):
    x_ = [som_b2.convert_map_to_euclidean(som_b2.winner(x))[0] + (2*np.random.rand(1)[0]-1)*0.4 
         for x in data.values[cluster_index==c]]
    y_ = [som_b2.convert_map_to_euclidean(som_b2.winner(x))[1] + (2*np.random.rand(1)[0]-1)*0.4 
         for x in data.values[cluster_index==c]]
    y_ = [(i*2 / np.sqrt(3) * 3 / 4) for i in y_]
    plot = sns.scatterplot(x=x_, y=y_, label='cluster='+str(c), alpha=.7)

xrange = np.arange(weights.shape[0])
yrange = np.arange(weights.shape[1])
plot = plt.xticks(xrange-.5, xrange)
plot = plt.yticks(yrange * 2 / np.sqrt(3) * 3 / 4, yrange)

divider = make_axes_locatable(plt.gca())
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
cb1 = colorbar.ColorbarBase(ax_cb, cmap=cm.Blues,
                           orientation='vertical', alpha=.4)

cb1.ax.get_yaxis().labelpad = 16
plot = cb1.ax.set_ylabel('distance from neurons in the neighborhood',
                        rotation=270, fontsize=16)
plot = plt.gcf().add_axes(ax_cb)

plt.savefig('som_visualization.png')


# In[45]:


cnt = []
for c in np.unique(cluster_index):
    count_c = len(data[cluster_index==c])
    cnt.append([c, count_c])
    
df_cnt = pd.DataFrame(cnt, columns=['cluster이름', '개수'])
df_cnt


# In[46]:


cluster_index


# In[59]:


df_river = pd.read_csv('새만금_하천_오염도.csv')
df_river['Cluster'] = cluster_index
df_river[df_river['Cluster'] == 3]


# In[56]:


df_river.to_csv('새만금_하천_Cluster.csv')


# In[ ]:




