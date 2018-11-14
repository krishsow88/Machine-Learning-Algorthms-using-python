# Market Basket Analysis
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
data.info()
data.columns
print("List of Items sold at the Bakery:")
print("Total Items: ",len(data.Item.unique()))
print("-"*15)
for i in data.Item.unique():
    print(i)


data.describe(include='all')
len(data.loc[data["Item"] == "NONE",:])
data.loc[data["Item"] == "NONE",:].head(10)
data.loc[data["Item"] == "NONE",:].tail(10)
data["Item"].value_counts().head(15)
# Pie Chart
plt.figure(1, figsize=(10,10))
data['Item'].value_counts().head(15).plot.pie(autopct="%1.1f%%")
plt.show()

itemNames = data['Item'].value_counts().index
itemValues = data['Item'].value_counts().values

plt.figure(figsize=(12,12))
plt.ylabel('Values', fontsize='medium')
plt.xlabel('Items', fontsize='medium')
plt.title('Top 20 Sell Bakery Items')
plt.bar(itemNames[:10],itemValues[:10], width = 0.7, color="blue",linewidth=0.4)
plt.show()


fig, ax = plt.subplots(figsize=(12,12))
plt.style.use('fivethirtyeight')
ax.barh(itemNames[:5], itemValues[:5])
plt.show()


firstMorning = data.loc[(data['Time']>='06:00:00')&(data['Time']<'09:00:00')]
secondMorning = data.loc[(data['Time']>='09:00:00')&(data['Time']<'12:00:00')]
firstAfternoon = data.loc[(data['Time']>='12:00:00')&(data['Time']<'15:00:00')]
secondAfternoon = data.loc[(data['Time']>='15:00:00')&(data['Time']<'18:00:00')]
night = data.loc[(data['Time']>='18:00:00')&(data['Time']<'21:00:00')]
hourlySales = {'firstMorning': len(firstMorning), 'secondMorning': len(secondMorning), 'firstAfternoon': len(firstAfternoon),'secondAfternoon': len(secondAfternoon),'night': len(night)}
print("This is night sales: ", hourlySales['night'])


fig, ax = plt.subplots(figsize=(10,10))
ax.barh(range(len(hourlySales)), list(hourlySales.values()), align='center')
plt.show()

print(firstMorning['Item'].value_counts().head(15))

# Bar Plot
plt.figure(figsize=(10,10))
plt.ylabel('Values', fontsize='medium')
plt.xlabel('Items', fontsize='medium')
plt.title('Top 20 Sell Bakery Items')
plt.bar(firstMorning['Item'][:5],firstMorning['Item'].value_counts()[:5], width = 0.7, color="blue",linewidth=0.4)
plt.show()

data['datetime'] = pd.to_datetime(data['Date']+" "+data['Time'])
data['year'] = data['datetime'].dt.year
data['month'] = data['datetime'].dt.month
data['day'] = data['datetime'].dt.day
data['weekday'] = data['datetime'].dt.weekday
data['hour'] = data['datetime'].dt.hour
data = data.drop(['Date'], axis=1)

data.head(5)

yearlyTable = data['year'].value_counts().plot(kind='bar',figsize=(10,5))
yearlyTable.set_xlabel("")
data['year'].value_counts().head()


data['monthlyTransaction'] = pd.to_datetime(data['datetime']).dt.to_period('M')
monthlyTransaction = data[['monthlyTransaction','Transaction']].groupby(['monthlyTransaction'], as_index=False).count().sort_values(by='monthlyTransaction')
monthlyTransaction.set_index('monthlyTransaction' ,inplace=True)

monthlyTable = monthlyTransaction.plot(kind='bar',figsize=(10,6))
monthlyTable.set_xlabel("")

monthlyTransaction

hourlyTransaction = data[['hour','Transaction']].groupby(['hour'], as_index=False).count()
hourlyTransaction.head(10)
hourlyTransaction.set_index('hour' ,inplace=True)

tableSort = hourlyTransaction.plot(kind='bar',figsize=(10,6))
tableSort.set_xlabel("")

hourlyTransaction

data['monthly'] = pd.to_datetime(data['datetime']).dt.to_period('M')
monthlyTransactionForItem = data[['monthly','Transaction', 'Item']].groupby(['monthly', 'Item'], as_index=False).count().sort_values(by='monthly')
monthlyTransactionForItem.set_index('monthly' ,inplace=True)

monthlyTransactionForItem.head(35)


cofeeSalesMonthly = monthlyTransactionForItem[monthlyTransactionForItem['Item']=='Coffee'].plot(kind='bar', figsize=(10,6))
cofeeSalesMonthly.set_xlabel("Coffee Sales Monthly")

plt.ylabel('Transaction', fontsize=16)
plt.xlabel('Month', fontsize=16)
plt.title("Monthly Coffee Sales", fontsize=16);

data['daily'] = pd.to_datetime(data['datetime']).dt.to_period('D')
dailyTransactionForItem = data[['daily','Transaction', 'Item']].groupby(['daily', 'Item'], as_index=False).count().sort_values(by='daily')
dailyTransactionForItem.set_index('daily' ,inplace=True)

dailyTransactionForItem.head(35)

data['hourly'] = pd.to_datetime(data['datetime']).dt.to_period('H')
hourlyTransactionForItem = data[['hourly','Transaction', 'Item']].groupby(['hourly', 'Item'], as_index=False).count().sort_values(by='hourly')
hourlyTransactionForItem.set_index('hourly' ,inplace=True)

hourlyTransactionForItem.head(35)





