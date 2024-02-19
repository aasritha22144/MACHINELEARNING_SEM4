import statistics
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns

#Taking dataset from excel sheet 
data_set = pd.read_excel(r"C:\Users\aasri\Downloads\Lab Session1 Data.xlsx", sheet_name="IRCTC Stock Price")

# for mean and variiable 
col_D = data_set.iloc[:,3]
stat_mean = statistics.mean(col_D)
stat_var = statistics.variance(col_D)

# For wednesday mean calculation 
data_set["Date"] = pd.to_datetime(data_set["Date"])
data_set['weekday'] = data_set["Date"].dt.weekday
wednesdays = data_set[data_set['weekday'] == 2]
wednesdays_price = wednesdays['Price']
wednesday_mean = wednesdays_price.mean()

data_set['Month'] = data_set["Date"].dt.month
april_data = data_set[data_set["Month"]==4]
April_mean = statistics.mean(april_data['Price'])

#probability for making lose in the stock
is_loss = lambda x:x<0 
loss_count = data_set['Chg%'].apply(is_loss).sum()
total_count = len(data_set)
probability_of_loss = loss_count/total_count

#probability of making a profit on wednesdays 
is_profit = lambda x:x>0
profit_count = data_set[data_set['weekday'] == 2]['Chg%'].apply(is_profit).sum()
probability_of_profit_wed = profit_count/total_count

# Calculate the conditional probability of making profit, given that today is Wednesday
profitable_wednesday = wednesdays[wednesdays['Chg%'] > 0]
num_profitable_wednesdays = len(profitable_wednesday)
total_wed = len(wednesdays)
conditional_prob_wed = num_profitable_wednesdays/total_wed

# scatter-plot of Chg% aganist the day of week
data_set["Day_of_week"] = data_set['Date'].dt.weekday
sns.scatterplot(x="Day_of_week", y="Chg%", data=data_set, hue="Day_of_week", palette="hls")
plt.xlabel("Day of the Week")
plt.ylabel("Chg%")
plt.title("Chg% Distribution by Day of the Week")
plt.show()

print(f"the mean of the wwednesday is {wednesday_mean}")
print(f"The mean of the dataset is {stat_mean}")
print(f"The variance of the dataset is {stat_var}")
print(f"the mean of the April month is {April_mean}")
print(f"The probability Making the loss of the stock {probability_of_loss}")
print(f"the probability of profit making on wednesday is {probability_of_profit_wed}")
print(f"The conditional probability of making profit that given date is wednes day is : {conditional_prob_wed}")