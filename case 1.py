#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv(r'C:\Users\huang\stout interview\loans_full_schema.csv')


# ## Description and Assumptions. Problem statement
# 
# This data set represents thousands of loans made through the Lending Club platform. Some loans are safer and some aren't. The data represents loans made and not loan applications. The dataset favors less risky.
# 
# Higher risk- high interest rate. low risk- low interest rate.
# 
# Problem statement- using data, create a model to predict the interest rate

# In[3]:


df.head(10)


# The categorical variables emp_title, state, homeownership, verified income, verification_income_ joint, loan_purpose, application_type, grade, sub_grade, issue_month, loan_status, initial_listing_status, disbursement_method
# 
# Split into segments-
# joint or individual
# 
# Expectations on interest rate:
# emp-title : higher pay jobs( supervisors) -- lower interest rates
# emp-length: longer length -- lower interest rate
# state: none
# homeownership: mortgage (higher credit rating) / OWN -- lower interest
# 
# annual_income: higher pay -- lower interest
# verified income: verified -- lower interest
# debt_to_income: low -- lower interest
# 
# annual_income_joint: higher pay -- lower interest
# verif_income_joints: verified -- lower interest
# debt_to_income_joint: low -- lower interest
# 
# subgrade: A -- lower interest rate
# issue_month: none
# loan_status: none
# initial_listing_status: none
# disbursement method: none
# balance: high -- lower interest rate
# paid total: high --low interest
# paid principal: high --low interest
# paid rate: high --low interest
# paid rate: high --low interest
# paid_late_fee: low -- low interest

# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df


# ## Find the distribution of the label attribute- interest_rate

# In[7]:


#descriptive statistics summary
df['interest_rate'].describe()


# In[8]:


sns.displot(df['interest_rate'], kde='true');


# The graph is skewed to the right. Since the graph is skewed, we should use a robust measure of the center . The median is 11.98. 

# ## Find correlation between numerical attributes and interest_rate and among numerical attributes with each other
# Get the corr_matrix.

# In[9]:


corr_matrix = df.corr()
corr_matrix

plt.figure(figsize = (30,30))
sns.heatmap(corr_matrix, annot=True, cmap='Blues')


# We focus on the top right half of the heatmap.The open_credit_lines and num_satisfactory are perfectly correlated. We might have to combine the attributes that are highly correlated. They might be attributes that are measuring the same thing from a different angle. An example is loan amount and installments are correlated, which makes sense because the more you own, the more you have to pay back. Another example is paid_total and paid_principle. If you paid a lot, you are paying down the principle and not the interest.

# In[10]:


corr_matrix ["interest_rate"].sort_values(ascending=False)


#  Which numerical have positive/negative assoication with "interest rate"?
# 
# Positive- Paid_interest, term, debt_to_income_joint, debt_to_income
# 
# Negative- total_debit_limit, annual_income_joint,num_mort_account, total_credit_limit, account_never_delinq_percent, months_since_last_inquiry
# 
# Then plot the attributes with highest/lowest correlation to see the type of relationship with "interest rate" and each other- linear

# The attribute "term" consist of discrete values 36 and 60- so we don't include that in scatterplot

# In[11]:


df["term"]


# In[12]:


from pandas.plotting import scatter_matrix

attributes = ["interest_rate", "paid_interest", "debt_to_income_joint", "annual_income_joint","total_debit_limit"]

scatter_matrix(df[attributes], figsize=(10, 10))


# We focus on the left most column.
# 
# The scatterplots shows that the paid_interest becomes more disperse as the interest rate increases- more risky. However, there seems to be a line like a cutoff.
# 
# The debt_to_income shows high variation but there is a positive correlation.
# 
# For annual_income_joint and total_income_joint, as interest rates increases those attributes become less disperse and concentrate to lower annual income and lower debit_limit.
# 
# Conclusion: Higher interest rates- less dispersion. Lower_income and lower_debit_limit means more risky- higher interest rates. Higher paid_interest and debt_to_income more risky- highest interest rate. These factors make sense.

# In[13]:


from pandas.plotting import scatter_matrix

attributes = ["interest_rate", "paid_interest", "debt_to_income_joint", "annual_income_joint","total_debit_limit", 'application_type']

sns.pairplot(df[attributes], hue='application_type')


# When we add the application type to the equation. For paid_interest and total debt limit, the application type have no difference in the interest rate. 

# ## Categorical Variables
# 
# Boxplots tell us the 25%, 50%, 75% quantile. The whisker measures the spread. 

# In[14]:


sns.boxplot(x='homeownership', y='interest_rate', data= df, hue='application_type')


# The type of homeownership have no impact on the interest_rate.

# In[15]:


sns.boxplot(x='verified_income', y='interest_rate', data= df, hue='application_type')


# Maybe verification of income does not have large impact on interest_rate. All the median interest_rate is from 10-15, the variance is also large, so the difference in median is not significant. 

# In[16]:


sns.boxplot(x='grade', y='interest_rate', data= df, hue='application_type' ,order=['A', 'B', 'C', 'D', 'E', 'F', 'G'])


# The grade associated with the loan affects the interest rate- lower lexicographical order the lower the interest rate. There is no difference between the different application types. There is also an outlier for grade D.

# ## Cleaning Data

# I will drop the rows with na. I can also fill the na with values but that would require business knowledge

# In[17]:


df = df.dropna(subset=["emp_title"])
df.info() 


# The months_since_last_delinq, months_since_90d_late, months_since_last_credit_inquiry, num_accounts_120d_past_due contains a lot of na values. I am not sure if na stands for 0. I will remove those colums just in case.
# 
# The debt_to_income has one null so I will just delete that row.
# 
# I won't remove annual_income_joint, verification_income_joint, debt_to_income_joint because na means individual. The # of non-null are not same across the rows. Just in case, I will delete the na rows whose application type is not individual.

# In[18]:


df = df[~(df['verification_income_joint'].isna() & (df.application_type =='joint'))]
df = df.dropna(subset=["debt_to_income"])
df = df.drop(['months_since_last_delinq', 'months_since_90d_late', 'num_accounts_120d_past_due', 'months_since_last_credit_inquiry'], axis=1)


# I will separate the loans that are joint and those that are individual. I think it is a good idea to train a different model for each application type since the populations are not similar otherwise.

# In[19]:


df_indiv = df[(df.application_type =='individual')]


# In[20]:


df_indiv = df_indiv.drop(['annual_income_joint','verification_income_joint','debt_to_income_joint'],axis=1)


# In[21]:


df_joint = df[(df.application_type =='joint')]


# Use an encoder to convert the labels into numerical form so it is machine readable

# In[22]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df_indiv = df_indiv.apply(encoder.fit_transform)
df_indiv


# In[23]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df_joint = df_joint.apply(encoder.fit_transform)
df_joint


# In[24]:


df_indiv.drop('emp_title', axis=1)


# Splitting the data into train feature, train level, test feature, test label

# In[25]:


X = df_indiv.drop("interest_rate", axis=1)
y = df_indiv["interest_rate"].copy()


# In[26]:


X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=.2, random_state=42)

X_train.shape , X_test.shape, y_train.shape , y_test.shape


# ## Model
# 
# We need regression model to predict data. Supervised learning algorithm.
# 
# We will use linear regression and logistic regression

# In[27]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_log = logreg.predict(X_test)


# In[28]:


# Linear Regression

from sklearn.linear_model import LinearRegression

linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred_lin = linreg.predict(X_test)


# In[29]:


# Log Regression

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math

MSE_log = mean_squared_error(y_test, y_pred_log)
RMSE_log = math.sqrt(mean_squared_error(y_test, y_pred_log))
MAE_log = mean_absolute_error(y_test, y_pred_log)


# In[30]:


# Linear Regression

MSE_lin = mean_squared_error(y_test, y_pred_lin)
RMSE_lin = math.sqrt(mean_squared_error(y_test, y_pred_lin))
MAE_lin = mean_absolute_error(y_test, y_pred_lin)


# ## Conclusion:
# 
# The MSE, RMSE, MAE is higher for logistic regression. This might be because the model iterations limit was reached and it did not fully converge to a solution. The model is able to generalize the data well. Next time, I will try more models and see if I can lower the error. Next time, I would also create a model for the joint application type.

# In[31]:


models = pd.DataFrame({
    'Model': ['Linear Regression', 'Logistic Regression'],
    'MSE': [MSE_lin, MSE_log],
    'RMSE': [RMSE_lin, RMSE_log],
    'MAE': [MAE_lin, MAE_log]})
models


# In[ ]:




