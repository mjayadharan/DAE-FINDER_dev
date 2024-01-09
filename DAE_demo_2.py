#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.integrate import odeint
import pandas as pd
import warnings
pd.set_option('display.float_format', '{:0.5f}'.format)



k_rates_df = pd.read_csv("parameters/k_rates_param.csv")


# In[3]:


k_rates_df


# #### Loading params

# In[4]:


IC_df = pd.read_csv("parameters/init_cond.csv")


# In[5]:


IC_df


# In[6]:


from dae_finder import solveMM, plotToy_MM, plotToyEnz


# #### Plotting solutions to different IC

# In[7]:


refinement_level = 2
final_time = 4
time_grid_1 = np.linspace(0, final_time, 10**2)
data_matrix_df_list = []

for k_rate_index in k_rates_df.index:
    k_rates = k_rates_df.loc[k_rate_index]
    for IC_index in IC_df.index:
        IC = IC_df.loc[IC_index]
        # data_matrix_df_list.append(solveMM(IC, k_rates, time_grid_1, "dense"))
        sol = solveMM(IC, k_rates, time_grid_1, "dense")
        sol_df = pd.DataFrame(sol, columns=IC.keys())
        data_matrix_df_list.append(sol_df)

data_matrix_df = pd.concat(data_matrix_df_list, ignore_index=True)
data_matrix_df



for sol_ in data_matrix_df_list:
    plotToyEnz(time_grid_1, sol_.to_numpy(), "MM QSSA approx: ") 
    



from dae_finder import PolyFeatureMatrix

poly_feature_ob = PolyFeatureMatrix(2)

candidate_lib_full = poly_feature_ob.fit_transform(data_matrix_df)


# In[14]:


candidate_lib_full




candidate_lib_full.describe()


# #### Splitting to training and test set in the case of mse-validation appraoch



test_set_index = int(0.9*len(candidate_lib_full))
candidate_lib = candidate_lib_full[:test_set_index]
candidate_lib_test = candidate_lib_full[test_set_index:]



from dae_finder import AlgModelFinder


# In[18]:


#Use lasso model by default
algebraic_model_1 = AlgModelFinder()

algebraic_model_1.fit(candidate_lib, scale_columns= True)


# In[19]:


#Best 10 models using R2 metrix
algebraic_model_1.best_models(5)


# ### Using different metric for selection

# In[25]:


algebraic_model_1.best_models(X_test=candidate_lib_test, metric="mse")


# In[ ]:





# ### Using Ridge regression to fit library terms

# In[26]:


#Using ridge regression to fit library terms
algebraic_model_2 = AlgModelFinder(model_id='RR')

algebraic_model_2.fit(candidate_lib, scale_columns= True)


# In[27]:


#Best 10 models using R2 metrix
algebraic_model_2.best_models(num=3)


# ### Using OLS to fit library terms

# In[28]:


#Using ridge regression to fit library terms
algebraic_model_3 = AlgModelFinder(model_id='LR')

algebraic_model_3.fit(candidate_lib, scale_columns= True)


# In[29]:


#Best 10 models using R2 metrix
algebraic_model_3.best_models(num=10)


# In[ ]:





# ## Sequential Thresholding Class

from dae_finder import sequentialThLin


# In[31]:


# scaling data
from sklearn.preprocessing import StandardScaler
s_scaler = StandardScaler(with_std=True, with_mean=False)
scaled_cand_lib = pd.DataFrame(s_scaler.fit_transform(candidate_lib), columns=s_scaler.feature_names_in_)
scaled_cand_lib['1'] = 1

feature = "S E"
X_ = scaled_cand_lib.drop([feature], axis=1)
y_ = scaled_cand_lib[feature]


# In[58]:


seq_model_1 = sequentialThLin(model_id="RR", alpha=0.05 , coef_threshold=0.2)
seq_model_4 = sequentialThLin()
seq_model_1.fit(X_,y_)
seq_model_4.fit(X_,y_)


# In[59]:


seq_model_1.coef_history_df_pre_thesh


# In[60]:


seq_model_1.coef_history_df


# In[61]:


seq_model_1.coef_


# ### Using Elastic net for thresholding

# In[54]:


seq_model_2 = sequentialThLin(model_id="EN", l1_ratio= 0.1, alpha=0.05 , coef_threshold=0.2)

seq_model_2.fit(X_,y_)


# In[55]:


seq_model_2.coef_history_df


# ## Using the thresholding model as a custom model in Algebraic model selector class

# In[56]:


algebraic_model_Th = AlgModelFinder(custom_model= True, custom_model_ob= seq_model_1)

algebraic_model_Th.fit(candidate_lib, scale_columns= False)


# In[57]:


#Best 10 models using R2 metrix
algebraic_model_Th.best_models()


# In[44]:


algebraic_model_Th.r2_score_dict

