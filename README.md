# CLTV Prediction with BG-NBD and Gamma-Gamma using [FLO](https://www.flo.com.tr/)'s Dataset

This is the CLTV Prediction _Real Life Case Study_ of [Miuul](https://miuul.com/) Data Science & Machine Learning
Bootcamp

## About FLO

---
Flo is a big Turkish fashion retail company, which has been operating since 2010. It offers various
products
such as shoes, bags, and clothes for both men and women. Flo has an extensive network of stores across Turkey and has
recently expanded its operations to other countries.

## Business Problem

---

FLO wants to set a roadmap for sales and marketing activities.
In order for the company to make a medium-long-term plan, it is necessary to estimate the potential value that existing
customers will provide to the company in the future.

## Story of Dataset

---

The dataset consists of information obtained from the past shopping behaviors of
customers who made their last purchases as OmniChannel (both online and offline shopper) in 2020 - 2021.

- master_id: Unique customer number

- order_channel : Which channel of the shopping platform is used (Android, ios, Desktop, Mobile, Offline)

- last_order_channel : The channel where the most recent purchase was made

- first_order_date : Date of the customer's first purchase

- last_order_date : Customer's last purchase date

- last_order_date_online : The date of the last purchase made by the customer on the online platform

- last_order_date_offline : Last shopping date made by the customer on the offline platform

- order_num_total_ever_online : The total number of purchases made by the customer on the online platform

- order_num_total_ever_offline : Total number of purchases made by the customer offline

- customer_value_total_ever_offline : Total fee paid by the customer for offline purchases

- customer_value_total_ever_online : The total fee paid by the customer for their online shopping

- interested_in_categories_12 : List of categories the customer has shopped in the last 12 months

---

## TASKS

### TASK 1: Data Preparation

1. Read the flo_data_20K.csv data. Make a copy of the dataframe.

2. Define the outlier_thresholds and replace_with_thresholds functions needed to suppress outliers.

3. Suppress "order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "
   customer_value_total_ever_online" if there are outliers.
4. Omnichannel means that customers shop from both online and offline platforms. Create new variables for each
   customer's total purchases and spend.
5. Examine the variable types. Change the type of variables that express date to date.

### TASK 2: Creating the CLTV Data Structure

1. Take 2 days after the date of the last purchase in the data set as the date of analysis.
2. Create a new cltv dataframe with customer_id, recency_cltv_weekly, T_weekly, frequency and monetary_cltv_avg values.
   Monetary value will be expressed as the average value per purchase, and recency and tenure values will be expressed
   in weekly terms.

### TASK 3: BG/NBD, Establishment of Gamma-Gamma Models, Calculation of CLTV

1. Please fit the BG/NBD model.
    1. Estimate expected purchases from customers in 3 months and add exp_sales_3_month to cltv dataframe.
    2. Estimate expected purchases from customers in 6 months and add exp_sales_6_month to cltv dataframe.
2. Fit the Gamma-Gamma model. Estimate the average value of the customers and add it to the cltv dataframe as
   exp_average_value.
3. Calculate 6 months CLTV and add it to the dataframe with the name cltv.
    1. Observe the 20 people with the highest Cltv value.

### TASK 4: Creating Segments by CLTV

1. Divide all your 6 month old customers into 4 groups (segments) and add the group names to the
   data set. Add it to the dataframe with the name cltv_segment.
2. Make short 6-month action suggestions to the management for 2 groups
   that you will choose from among 4 groups.

### BONUS: Functioning the whole process.

Write a function that does whole the process
