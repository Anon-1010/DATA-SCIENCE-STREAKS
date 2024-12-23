#!/usr/bin/env python
# coding: utf-8

# # SALES OF A SUPERMARKET

# ### i) IMPORTING NECESSARY LIBRARIES

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[7]:


df=pd.read_csv('supermarket_sales.csv') 


# In[6]:


df.head()


# ### ii) DATA CLEANING

# In[11]:


df.info()


# In[13]:


df.isnull().sum()


# #### Conclusion - There's no null value present in this dataset

# In[15]:





# In[29]:


#Standardize Categorical Data - Ensure consistency in categorical values (e.g., avoid "Male" and "male").
df['Gender'] = df['Gender'].str.capitalize()
df['Customer type'] = df['Customer type'].str.capitalize()
df['City'] = df['City'].str.strip()  # Remove leading/trailing spaces


# In[26]:


print(df.dtypes)  #To check the datatype of the dataset


# In[22]:


df.describe()


# ### iii) Generating Descriptive Statistics

# In[31]:


# Average unit prize
avg_unit_price = df['Unit price'].mean()
print(f"Average Unit Price: {avg_unit_price:.2f}")


# In[33]:


# Total Sales Per Branch
total_sales_branch = df.groupby('Branch')['Total'].sum()
print("Total Sales Per Branch:")
print(total_sales_branch)


# In[34]:


# Average Rating Per Product Line
avg_rating_product_line = df.groupby('Product line')['Rating'].mean()
print("Average Rating Per Product Line:")
print(avg_rating_product_line)


# In[36]:


# Total Quantity Sold Per City
total_quantity_city = df.groupby('City')['Quantity'].sum()
print("Total Quantity Sold Per City:")
print(total_quantity_city)


# In[38]:


# Top Payment Method
top_payment_method = df['Payment'].value_counts().idxmax()
print(f"Top Payment Method: {top_payment_method}")


# In[39]:


# Gross Income Per Branch:
gross_income_branch = df.groupby('Branch')['gross income'].sum()
print("Gross Income Per Branch:")
print(gross_income_branch)


# In[41]:


# Average Sales by Customer Type:
avg_sales_customer_type = df.groupby('Customer type')['Total'].mean()
print("Average Sales by Customer Type:")
print(avg_sales_customer_type)


# In[75]:


# Most Popular Product Line:
popular_product_line = df['Product line'].value_counts().idxmax()
print(f"Most Popular Product Line: {popular_product_line}")


# ## iii) Exploratory Data Analysis (EDA)

# #### Bar plot to show which branch has the highest sales records.

# In[74]:


data['Branch'].value_counts().sort_values(ascending=False).plot(kind='bar',figsize=(10,5),color=['orange','blue','green'])
plt.title('Number of records in each brachs')
plt.xlabel('Branch')
plt.ylabel('Count of values')
plt.show()


# #### Pie chart to visualize the gender distribution in the data.

# In[76]:


data['Gender'].value_counts().sort_values(ascending=False).plot(kind='pie',figsize=(10,5),labels=['Female','Male'], colors=['#1b4965','#f35b04'], autopct='%1.2f%%', shadow=True)
plt.title('Gener distribution')
plt.legend()
plt.show()


# #### Trends Analysis: Ploting total sales by month and branch.

# In[77]:


df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.to_period('M')


# In[55]:


# Aggregating sales data by Month and Branch
sales_trends = df.groupby(['Month', 'Branch'])['Total'].sum().reset_index()
# Converting 'Month' to datetime format for plotting
sales_trends['Month'] = sales_trends['Month'].dt.to_timestamp()


# In[57]:


plt.figure(figsize=(12, 6))
sns.lineplot(data=sales_trends, x='Month', y='Total', hue='Branch', marker='o', palette='viridis')
plt.title('Total Sales by Month and Branch', fontsize=14)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Total Sales', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Branch')
plt.grid(True)
plt.show()


# '''Conclusion  From the analysis, it is clear that January 1 has the highest sales compared to all other days in the dataset. This could be due to: 
# New Year's Celebrations: People might be shopping for celebrations, gifts, or other holiday-related needs on this day.
# Special Offers: Businesses might run New Year discounts or promotions, leading to higher sales.
# Increased Customer Activity: Many people shop more during holidays, which could explain the spike in sales.
# This shows how important New Year’s Day is for businesses. Companies can take advantage of this trend by planning special campaigns or offers to boost sales during similar occasions.'''

# #### Total Quantity Sold by Product Category

# In[78]:


quantity_by_category = df.groupby('Product line')['Quantity'].sum().sort_values(ascending=False)
print(quantity_by_category)


# In[69]:


#Plotting Quantity Sold by Product Category
plt.figure(figsize=(10, 6))
quantity_by_category.plot(kind='bar', color='skyblue')
plt.title('Total Quantity Sold by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Total Quantity Sold')
plt.xticks(rotation=45)
plt.show()


# #### Total Revenue by Product Category

# In[79]:


revenue_by_category = df.groupby('Product line')['Total'].sum().sort_values(ascending=False)
print(revenue_by_category)


# In[73]:


# Plotting Revenue by Product Category
plt.figure(figsize=(10, 6))
revenue_by_category.plot(kind='bar', color='lightgreen')
plt.title('Total Revenue by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Total Revenue')
plt.xticks(rotation=45)
plt.show()


# In[59]:


# Group by Branch and Payment method to calculate the count of each payment method per branch
payment_method_analysis = df.groupby(['Branch', 'Payment']).size().unstack().fillna(0)
print(payment_method_analysis)


# In[80]:


payment_method_analysis.plot(kind='bar', stacked=True, figsize=(12, 6), color=['#8B0000', '#32CD32', '#1E90FF', '#FFD700'])
plt.title('Payment Method Popularity Across Branches', fontsize=14)
plt.xlabel('Branch', fontsize=12)
plt.ylabel('Number of Transactions', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Payment Method')
plt.show()


# In[61]:


payment_method_percentage = payment_method_analysis.div(payment_method_analysis.sum(axis=1), axis=0) * 100
print(payment_method_percentage)


# ## iv) Basic Predictive Modeling

# ### Converting categorical data 

# In[82]:


# One-hot encode categorical columns
df_encoded = pd.get_dummies(df, columns=['Product line', 'Branch', 'Payment', 'City', 'Customer type', 'Gender'], drop_first=True)


# In[83]:


# Define features (X) and target variable (y)
X = df_encoded[['Unit price', 'Quantity'] + list(df_encoded.columns[df_encoded.columns.str.contains('Product line|Branch|City|Customer type|Gender')])]
y = df_encoded['Total']


# In[84]:


from sklearn.model_selection import train_test_split
# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[85]:


from sklearn.linear_model import LinearRegression   #Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)


# In[86]:


from sklearn.tree import DecisionTreeRegressor             #Decision Tree
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)


# In[87]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Evaluating the Linear Regression model
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print(f"Linear Regression Model - MAE: {mae_lr}")
print(f"Linear Regression Model - MSE: {mse_lr}")
print(f"Linear Regression Model - R-squared: {r2_lr}")


# In[89]:


# Evaluating the Decision Tree model
mae_dt = mean_absolute_error(y_test, y_pred_dt)
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)
print(f"Decision Tree Model - MAE: {mae_dt}")
print(f"Decision Tree Model - MSE: {mse_dt}")
print(f"Decision Tree Model - R-squared: {r2_dt}")


# In[90]:


import matplotlib.pyplot as plt
# Plotting Actual vs Predicted values for Linear Regression
plt.scatter(y_test, y_pred_lr)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Line of perfect prediction
plt.xlabel('Actual Total Cost')
plt.ylabel('Predicted Total Cost')
plt.title('Actual vs Predicted Total Cost (Linear Regression)')
plt.show()


# In[91]:


# Plotting Actual vs Predicted values for Decision Tree
plt.scatter(y_test, y_pred_dt)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Line of perfect prediction
plt.xlabel('Actual Total Cost')
plt.ylabel('Predicted Total Cost')
plt.title('Actual vs Predicted Total Cost (Decision Tree)')
plt.show()


# In[92]:


'''Conclusion - Decision tree is more relable than the linear regression in this case'''


# # INSIGHTS

# In[94]:


'''Key Insight: Electronic Accessories as the Top-Selling Product Line
Popularity of Electronic Accessories: The data indicates that Electronic Accessories is the highest-performing product line, with 971 units sold, making it a significant revenue driver for the business. This highlights its strong demand among customers.'''


# In[95]:


'''Encourage Digital Payment Adoption: Given that cash is a dominant payment method, businesses can introduce incentives for customers who opt for digital payments (such as credit cards or mobile wallets). Offering a 5-10% discount or cashback on digital payments could motivate customers to use them more frequently. This will not only streamline payment processing but could also help businesses in managing cash flows and reducing the risk of cash handling.'''


# In[96]:


'''Seasonal Promotions: The seasonal trends (e.g., higher sales in January) suggest that customers may be more likely to shop during certain months, potentially due to events, holidays, or sales seasons. Businesses can plan seasonal campaigns around these high-sales months. For example, launching exclusive discounts or limited-time offers in January could align with the spike in sales, potentially boosting revenue further.'''


# In[ ]:




