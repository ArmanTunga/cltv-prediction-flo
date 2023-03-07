##############################################################
# CLTV Prediction with BG-NBD and Gamma-Gamma 
##############################################################
import pandas as pd

###############################################################
# Business Problem
###############################################################
# FLO wants to set a roadmap for sales and marketing activities.
# In order for the company to make a medium-long-term plan, it is necessary to estimate
# the potential value that existing customers will provide to the company in the future.


###############################################################
# Story of Dataset
###############################################################

# The dataset consists of information obtained from the past shopping behaviors of
# customers who made their last purchases as OmniChannel (both online and offline shopper) in 2020 - 2021.

# master_id: Unique customer number
# order_channel : Which channel of the shopping platform is used (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : The channel where the most recent purchase was made
# first_order_date : Date of the customer's first purchase
# last_order_date : Customer's last purchase date
# last_order_date_online : The date of the last purchase made by the customer on the online platform
# last_order_date_offline : Last shopping date made by the customer on the offline platform
# order_num_total_ever_online : The total number of purchases made by the customer on the online platform
# order_num_total_ever_offline : Total number of purchases made by the customer offline
# customer_value_total_ever_offline : Total fee paid by the customer for offline purchases
# customer_value_total_ever_online : The total fee paid by the customer for their online shopping
# interested_in_categories_12 : List of categories the customer has shopped in the last 12 months


###############################################################
# TASKS
###############################################################
# TASK 1: Data Preparation
# 1. Read the flo_data_20K.csv data. Make a copy of the dataframe.
# 2. Define the outlier_thresholds and replace_with_thresholds functions needed to suppress outliers.
# Note: When calculating cltv, the frequency values must be integers. Therefore, round the lower and upper limits with round().
# 3. Suppress "order_num_total_ever_online","order_num_total_ever_offline",
# "customer_value_total_ever_offline","customer_value_total_ever_online" if there are outliers.
# 4. Omnichannel means that customers shop from both online and offline platforms.
# Create new variables for each customer's total purchases and spend.
# 5. Examine the variable types. Change the type of variables that express date to date.

# TASK 2: Creating the CLTV Data Structure
# 1. Take 2 days after the date of the last purchase in the data set as the date of analysis.
# 2. Create a new cltv dataframe with customer_id, recency_cltv_weekly, T_weekly,
# frequency and monetary_cltv_avg values. Monetary value will be expressed as
# the average value per purchase, and recency and tenure values will be expressed in weekly terms.


# TASK 3: BG/NBD, Establishment of Gamma-Gamma Models, Calculation of CLTV
# 1. Please fit the BG/NBD model.
# a. Estimate expected purchases from customers in 3 months and add exp_sales_3_month to cltv dataframe.
# b. Estimate expected purchases from customers in 6 months and add exp_sales_6_month to cltv dataframe.
# 2. Fit the Gamma-Gamma model. Estimate the average value of the customers and add it to the cltv dataframe as exp_average_value.
# 3. Calculate 6 months CLTV and add it to the dataframe with the name cltv.
# b. Observe the 20 people with the highest Cltv value.

# TASK 4: Creating Segments by CLTV
# 1. Divide all your 6 month old customers into 4 groups (segments) and add the group names to the
# data set. Add it to the dataframe with the name cltv_segment.
# 2. Make short 6-month action suggestions to the management for 2 groups
# that you will choose from among 4 groups.

# BONUS: Functioning the Whole Process.


###############################################################
# TASK 1: Data Preparation
###############################################################
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 1000)

# 1. Read the OmniChannel.csv data. Make a copy of the dataframe.
df_ = pd.read_csv("datasets/flo_data_20k.csv")
df = df_.copy()
df.head()


# 2. Define the outlier_thresholds and replace_with_thresholds functions required to suppress outliers.
# Note: When calculating cltv, frequency values must be integers. Therefore,
# round the lower and upper limits with round().

def outlier_thresholds(dataframe: pd.DataFrame, variable: str) -> (float, float):
    """
    This func determines threshold value for the given variable of given dataframe

    Args:
        dataframe: Dataframe of given variable
        variable: Variable that we get threshold value from

    Returns:
        low_limit: Lower limit of threshold
         up_limit: Upper limit of threshold
    """
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit.round(), up_limit.round()


def replace_with_thresholds(dataframe: pd.DataFrame, variable: str):
    """
    This function modifies the given dataframe by replacing outlier values of
     given variable with the upper and lower limit threshold values
     Altough it effects the given dataframe and
     there is no need for returning the dataframe, I want to be able to
     assign updated dataframe with the old one to increase readability

    Args:
        dataframe: Dataframe that we want to get variable and modify
        variable: Variable that we want to replace values of outliers
    """
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
    return dataframe


# 3. Suppress "order_num_total_ever_online","order_num_total_ever_offline",
# "customer_value_total_ever_offline","customer_value_total_ever_online" if there are outliers.
df.describe().T
df = replace_with_thresholds(df, "customer_value_total_ever_offline")
df = replace_with_thresholds(df, "customer_value_total_ever_online")
# 4. Omnichannel means that customers shop from both online and offline platforms.
# Create new variables for each customer's total purchases and spend.
df["order_num_total"] = (df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]).astype(int)
df["customer_value_total"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]
df.head()
# 5. Examine the variable types. Change the type of variables that express date to date.
date_cols = [col for col in df.columns if "date" in col]
df[date_cols] = df[date_cols].astype("datetime64[ns]")

###############################################################
# TASK 2: Creating the CLTV Data Structure
###############################################################

# 1. Take 2 days after the date of the last purchase in the data set as the date of analysis.
last_order_date_max = df["last_order_date"].max()
last_order_date_max = dt.datetime.strptime(str(last_order_date_max), "%Y-%m-%d %H:%M:%S")

today_date = last_order_date_max + dt.timedelta(days=2)

# 2. Create a new cltv dataframe with customer_id, recency_cltv_weekly, T_weekly, frequency and monetary_cltv_avg values.
df["master_id"].nunique() == df.shape[0]

cltv_df = df.groupby('master_id').agg(
    {"last_order_date": lambda last_order_date: last_order_date.iloc[0],
     "first_order_date": lambda first_order_date: first_order_date.iloc[0],
     "order_num_total": lambda order_num_total: order_num_total,
     "customer_value_total": lambda customer_value_total: customer_value_total})

cltv_df["recency_cltv_weekly"] = (cltv_df["last_order_date"] - cltv_df["first_order_date"]).dt.days / 7
cltv_df["T_weekly"] = (today_date - cltv_df["first_order_date"]).dt.days / 7
cltv_df.head()
cltv_df = cltv_df.reset_index()

columns_to_drop = ["last_order_date", "first_order_date"]
cltv_df = cltv_df.drop(columns_to_drop, axis=1)
cltv_df.columns = ["master_id", "frequency", "monetary", "recency_cltv_weekly", "T_weekly"]
cltv_df.describe().T  # There is no frequency = 0 but let's include the line below for upcoming datasets
cltv_df = cltv_df[cltv_df["frequency"] > 0]
cltv_df["monetary_cltv_avg"] = cltv_df["monetary"] / cltv_df["frequency"]

###############################################################
# TASK 3: BG/NBD, Establishment of Gamma-Gamma Models, calculation of 6-month CLTV
###############################################################

# 1. Install the BG/NBD model.
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df["frequency"], cltv_df["recency_cltv_weekly"], cltv_df["T_weekly"])

# Estimate expected purchases from customers in 3 months and add exp_sales_3_month to cltv dataframe.
cltv_df["exp_sales_3_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4 * 3, cltv_df["frequency"],
                                                                                       cltv_df["recency_cltv_weekly"],
                                                                                       cltv_df["T_weekly"])

# Estimate expected purchases from customers in 6 months and add exp_sales_6_month to cltv dataframe.
cltv_df["exp_sales_6_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4 * 6, cltv_df["frequency"],
                                                                                       cltv_df["recency_cltv_weekly"],
                                                                                       cltv_df["T_weekly"])
# Please review the 10 people who will make the most purchases in the 3rd and 6th months.
cltv_df.loc[cltv_df["exp_sales_3_month"].sort_values(ascending=False).head(10).index]
cltv_df.loc[cltv_df["exp_sales_6_month"].sort_values(ascending=False).head(10).index]
plot_period_transactions(bgf)
plt.show()
# 2. Fit the Gamma-Gamma model.

ggf = GammaGammaFitter(penalizer_coef=.01)
ggf.fit(cltv_df["frequency"], cltv_df["monetary_cltv_avg"])
# Estimate the average value of the customers and add it to the cltv dataframe as exp_average_value.
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                       cltv_df['monetary_cltv_avg'])

# 3. Calculate 6 months CLTV and add it to the dataframe with the name cltv.
cltv_df['cltv'] = ggf.customer_lifetime_value(bgf,
                                              cltv_df['frequency'],
                                              cltv_df['recency_cltv_weekly'],
                                              cltv_df['T_weekly'],
                                              cltv_df['monetary_cltv_avg'],
                                              time=6,
                                              freq="W",
                                              discount_rate=0.01)

# Observe the 20 people with the highest CLTV.
cltv_df.sort_values("cltv", ascending=False).head(20)

###############################################################
# TASK 4: Creating Segments by CLTV
###############################################################

# 1. Divide all your customers into 4 groups (segments)
# according to 6-month CLTV and add the group names to the dataset. Assign it with the name cltv_segment.
cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])
cltv_df.head()
# 2. Examine the recency, frequency and monetary averages of the segments.
cltv_df.groupby("cltv_segment").agg({"count", "mean", "sum"})
