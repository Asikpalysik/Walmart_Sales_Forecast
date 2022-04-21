from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
import warnings
from arch.univariate import ARX, GARCH, StudentsT, Normal
from arch import arch_model
from pmdarima import auto_arima
from pmdarima.arima import decompose
from pmdarima.utils import decomposed_plot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn import preprocessing
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose as season
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import itertools
from datetime import timedelta
from datetime import datetime
import math
import seaborn as sns
import matplotlib as mpl
import numpy as np      # To use np.arrays
import pandas as pd     # To use dataframes
from pandas.plotting import autocorrelation_plot as auto_corr

# To plot
import matplotlib.pyplot as plt
%matplotlib inline

# For date-time

# Another imports if needs


warnings.filterwarnings("ignore")

pd.options.display.max_columns = 100  # to see columns

df_store = pd.read_csv(
    '/Users/asik/Desktop/Walmart_Sales_Forecast/Data/stores.csv')  # store data

df_train = pd.read_csv(
    '/Users/asik/Desktop/Walmart_Sales_Forecast/Data/train.csv')  # train set

df_features = pd.read_csv(
    '/Users/asik/Desktop/Walmart_Sales_Forecast/Data/features.csv')  # external information

df_store.head()

df_train.head()

df_features.head()

# merging 3 different sets
df = df_train.merge(df_features, on=['Store', 'Date'], how='inner').merge(
    df_store, on=['Store'], how='inner')
df.head(5)

df.drop(['IsHoliday_y'], axis=1, inplace=True)  # removing dublicated column

df.rename(columns={'IsHoliday_x': 'IsHoliday'},
          inplace=True)  # rename the column

df.head()  # last ready data set

df.shape

df['Store'].nunique()  # number of different values

df['Dept'].nunique()  # number of different values

store_dept_table = pd.pivot_table(df, index='Store', columns='Dept',
                                  values='Weekly_Sales', aggfunc=np.mean)
display(store_dept_table)

df.loc[df['Weekly_Sales'] <= 0]

df = df.loc[df['Weekly_Sales'] > 0]

df.shape  # new data shape

df['Date'].head(5).append(df['Date'].tail(5))  # to see first and last 5 rows.

sns.barplot(x='IsHoliday', y='Weekly_Sales', data=df)

df_holiday = df.loc[df['IsHoliday'] == True]
df_holiday['Date'].unique()

df_not_holiday = df.loc[df['IsHoliday'] == False]
df_not_holiday['Date'].nunique()

# Super bowl dates in train set
df.loc[(df['Date'] == '2010-02-12') | (df['Date'] == '2011-02-11')
       | (df['Date'] == '2012-02-10'), 'Super_Bowl'] = True
df.loc[(df['Date'] != '2010-02-12') & (df['Date'] != '2011-02-11')
       & (df['Date'] != '2012-02-10'), 'Super_Bowl'] = False

# Labor day dates in train set
df.loc[(df['Date'] == '2010-09-10') | (df['Date'] == '2011-09-09')
       | (df['Date'] == '2012-09-07'), 'Labor_Day'] = True
df.loc[(df['Date'] != '2010-09-10') & (df['Date'] != '2011-09-09')
       & (df['Date'] != '2012-09-07'), 'Labor_Day'] = False

# Thanksgiving dates in train set
df.loc[(df['Date'] == '2010-11-26') | (df['Date']
                                       == '2011-11-25'), 'Thanksgiving'] = True
df.loc[(df['Date'] != '2010-11-26') & (df['Date']
                                       != '2011-11-25'), 'Thanksgiving'] = False

# Christmas dates in train set
df.loc[(df['Date'] == '2010-12-31') |
       (df['Date'] == '2011-12-30'), 'Christmas'] = True
df.loc[(df['Date'] != '2010-12-31') &
       (df['Date'] != '2011-12-30'), 'Christmas'] = False

# Christmas holiday vs not-Christmas
sns.barplot(x='Christmas', y='Weekly_Sales', data=df)

# Thanksgiving holiday vs not-thanksgiving
sns.barplot(x='Thanksgiving', y='Weekly_Sales', data=df)

# Super bowl holiday vs not-super bowl
sns.barplot(x='Super_Bowl', y='Weekly_Sales', data=df)

# Labor day holiday vs not-labor day
sns.barplot(x='Labor_Day', y='Weekly_Sales', data=df)

# Avg weekly sales for types on Christmas
df.groupby(['Christmas', 'Type'])['Weekly_Sales'].mean()

# Avg weekly sales for types on Labor Day
df.groupby(['Labor_Day', 'Type'])['Weekly_Sales'].mean()

# Avg weekly sales for types on Thanksgiving
df.groupby(['Thanksgiving', 'Type'])['Weekly_Sales'].mean()

# Avg weekly sales for types on Super Bowl
df.groupby(['Super_Bowl', 'Type'])['Weekly_Sales'].mean()

my_data = [48.88, 37.77, 13.33]  # percentages
my_labels = 'Type A', 'Type B', 'Type C'  # labels
plt.pie(my_data, labels=my_labels, autopct='%1.1f%%', textprops={
        'fontsize': 15})  # plot pie type and bigger the labels
plt.axis('equal')
mpl.rcParams.update({'font.size': 20})  # bigger percentage labels

plt.show()

df.groupby('IsHoliday')['Weekly_Sales'].mean()

# Plotting avg wekkly sales according to holidays by types
plt.style.use('seaborn-poster')
labels = ['Thanksgiving', 'Super_Bowl', 'Labor_Day', 'Christmas']
A_means = [27397.77, 20612.75, 20004.26, 18310.16]
B_means = [18733.97, 12463.41, 12080.75, 11483.97]
C_means = [9696.56, 10179.27, 9893.45, 8031.52]

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize=(16, 8))
rects1 = ax.bar(x - width, A_means, width, label='Type_A')
rects2 = ax.bar(x, B_means, width, label='Type_B')
rects3 = ax.bar(x + width, C_means, width, label='Type_C')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Weekly Avg Sales')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.axhline(y=17094.30, color='r')  # holidays avg
plt.axhline(y=15952.82, color='green')  # not-holiday avg

fig.tight_layout()

plt.show()


df.sort_values(by='Weekly_Sales', ascending=False).head(5)


df_store.groupby('Type').describe()['Size'].round(
    2)  # See the Size-Type relation

plt.figure(figsize=(10, 8))  # To see the type-size relation
fig = sns.boxplot(x='Type', y='Size', data=df, showfliers=False)

df.isna().sum()

df = df.fillna(0)  # filling null's with 0

df.isna().sum()  # last null check

df.describe()  # to see weird statistical things

x = df['Dept']
y = df['Weekly_Sales']
plt.figure(figsize=(15, 5))
plt.title('Weekly Sales by Department')
plt.xlabel('Departments')
plt.ylabel('Weekly Sales')
plt.scatter(x, y)
plt.show()

plt.figure(figsize=(30, 10))
fig = sns.barplot(x='Dept', y='Weekly_Sales', data=df)

x = df['Store']
y = df['Weekly_Sales']
plt.figure(figsize=(15, 5))
plt.title('Weekly Sales by Store')
plt.xlabel('Stores')
plt.ylabel('Weekly Sales')
plt.scatter(x, y)
plt.show()

plt.figure(figsize=(20, 6))
fig = sns.barplot(x='Store', y='Weekly_Sales', data=df)

df["Date"] = pd.to_datetime(df["Date"])  # convert to datetime
df['week'] = df['Date'].dt.week
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year

df.groupby('month')['Weekly_Sales'].mean()  # to see the best months for sales

df.groupby('year')['Weekly_Sales'].mean()  # to see the best years for sales

monthly_sales = pd.pivot_table(
    df, values="Weekly_Sales", columns="year", index="month")
monthly_sales.plot()

fig = sns.barplot(x='month', y='Weekly_Sales', data=df)

df.groupby('week')['Weekly_Sales'].mean().sort_values(ascending=False).head()

weekly_sales = pd.pivot_table(
    df, values="Weekly_Sales", columns="year", index="week")
weekly_sales.plot()

plt.figure(figsize=(20, 6))
fig = sns.barplot(x='week', y='Weekly_Sales', data=df)

fuel_price = pd.pivot_table(df, values="Weekly_Sales", index="Fuel_Price")
fuel_price.plot()

temp = pd.pivot_table(df, values="Weekly_Sales", index="Temperature")
temp.plot()

CPI = pd.pivot_table(df, values="Weekly_Sales", index="CPI")
CPI.plot()

unemployment = pd.pivot_table(df, values="Weekly_Sales", index="Unemployment")
unemployment.plot()

# assign new data frame to csv for using after here
df.to_csv('clean_data.csv')

pd.options.display.max_columns = 100  # to see columns

df = pd.read_csv('/Users/asik/Desktop/Walmart_Sales_Forecast/clean_data.csv')

df.drop(columns=['Unnamed: 0'], inplace=True)

df['Date'] = pd.to_datetime(df['Date'])  # changing datetime to divide if needs

df_encoded = df.copy()  # to keep original dataframe taking copy of it

type_group = {'A': 1, 'B': 2, 'C': 3}  # changing A,B,C to 1-2-3
df_encoded['Type'] = df_encoded['Type'].replace(type_group)

df_encoded['Super_Bowl'] = df_encoded['Super_Bowl'].astype(
    bool).astype(int)  # changing T,F to 0-1

df_encoded['Thanksgiving'] = df_encoded['Thanksgiving'].astype(
    bool).astype(int)  # changing T,F to 0-1

df_encoded['Labor_Day'] = df_encoded['Labor_Day'].astype(
    bool).astype(int)  # changing T,F to 0-1

df_encoded['Christmas'] = df_encoded['Christmas'].astype(
    bool).astype(int)  # changing T,F to 0-1

df_encoded['IsHoliday'] = df_encoded['IsHoliday'].astype(
    bool).astype(int)  # changing T,F to 0-1

df_new = df_encoded.copy()  # taking the copy of encoded df to keep it original

drop_col = ['Super_Bowl', 'Labor_Day', 'Thanksgiving', 'Christmas']
df_new.drop(drop_col, axis=1, inplace=True)  # dropping columns

plt.figure(figsize=(12, 10))
sns.heatmap(df_new.corr().abs())    # To see the correlations
plt.show()

drop_col = ['Temperature', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment']
df_new.drop(drop_col, axis=1, inplace=True)  # dropping columns

plt.figure(figsize=(12, 10))
# To see the correlations without dropping columns
sns.heatmap(df_new.corr().abs())
plt.show()

# sorting according to date
df_new = df_new.sort_values(by='Date', ascending=True)

train_data = df_new[:int(0.7*(len(df_new)))]  # taking train part
test_data = df_new[int(0.7*(len(df_new))):]  # taking test part

target = "Weekly_Sales"
# all columns except weekly sales
used_cols = [c for c in df_new.columns.to_list() if c not in [target]]

X_train = train_data[used_cols]
X_test = test_data[used_cols]
y_train = train_data[target]
y_test = test_data[target]

X = df_new[used_cols]  # to keep train and test X values together

X_train = X_train.drop(['Date'], axis=1)  # dropping date from train
X_test = X_test.drop(['Date'], axis=1)  # dropping date from test


def wmae_test(test, pred):  # WMAE for test
    weights = X_test['IsHoliday'].apply(
        lambda is_holiday: 5 if is_holiday else 1)
    error = np.sum(weights * np.abs(test - pred), axis=0) / np.sum(weights)
    return error


rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=35,
                           max_features='sqrt', min_samples_split=10)

scaler = RobustScaler()


# making pipe tp use scaler and regressor together
pipe = make_pipeline(scaler, rf)

pipe.fit(X_train, y_train)

# predictions on train set
y_pred = pipe.predict(X_train)

# predictions on test set
y_pred_test = pipe.predict(X_test)

wmae_test(y_test, y_pred_test)

X = X.drop(['Date'], axis=1)  # dropping date column from X

importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Printing the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plotting the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

X1_train = X_train.drop(['month'], axis=1)  # dropping month
X1_test = X_test.drop(['month'], axis=1)

rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=35,
                           max_features='sqrt', min_samples_split=10)

scaler = RobustScaler()
pipe = make_pipeline(scaler, rf)

pipe.fit(X1_train, y_train)

# predictions on train set
y_pred = pipe.predict(X1_train)

# predictions on test set
y_pred_test = pipe.predict(X1_test)

wmae_test(y_test, y_pred_test)


# splitting train-test to whole dataset
train_data_enc = df_encoded[:int(0.7*(len(df_encoded)))]
test_data_enc = df_encoded[int(0.7*(len(df_encoded))):]

target = "Weekly_Sales"
used_cols1 = [c for c in df_encoded.columns.to_list(
) if c not in [target]]  # all columns except price

X_train_enc = train_data_enc[used_cols1]
X_test_enc = test_data_enc[used_cols1]
y_train_enc = train_data_enc[target]
y_test_enc = test_data_enc[target]

X_enc = df_encoded[used_cols1]  # to get together train,test splits

X_enc = X_enc.drop(['Date'], axis=1)  # dropping date column for whole X

# dropping date from train and test
X_train_enc = X_train_enc.drop(['Date'], axis=1)
X_test_enc = X_test_enc.drop(['Date'], axis=1)

rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=35,
                           max_features='sqrt', min_samples_split=10)

scaler = RobustScaler()
pipe = make_pipeline(scaler, rf)

pipe.fit(X_train_enc, y_train_enc)

# predictions on train set
y_pred_enc = pipe.predict(X_train_enc)

# predictions on test set
y_pred_test_enc = pipe.predict(X_test_enc)

wmae_test(y_test_enc, y_pred_test_enc)

importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Printing the feature ranking
print("Feature ranking:")

for f in range(X_enc.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plotting the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_enc.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(X_enc.shape[1]), indices)
plt.xlim([-1, X_enc.shape[1]])
plt.show()

# taking copy of encoded data to keep it without change.
df_encoded_new = df_encoded.copy()
df_encoded_new.drop(drop_col, axis=1, inplace=True)

# train-test splitting
train_data_enc_new = df_encoded_new[:int(0.7*(len(df_encoded_new)))]
test_data_enc_new = df_encoded_new[int(0.7*(len(df_encoded_new))):]

target = "Weekly_Sales"
used_cols2 = [c for c in df_encoded_new.columns.to_list() if c not in [
    target]]  # all columns except price

X_train_enc1 = train_data_enc_new[used_cols2]
X_test_enc1 = test_data_enc_new[used_cols2]
y_train_enc1 = train_data_enc_new[target]
y_test_enc1 = test_data_enc_new[target]

# droping date from train-test
X_train_enc1 = X_train_enc1.drop(['Date'], axis=1)
X_test_enc1 = X_test_enc1.drop(['Date'], axis=1)


rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=40,
                           max_features='log2', min_samples_split=10)

scaler = RobustScaler()
pipe = make_pipeline(scaler, rf)

pipe.fit(X_train_enc1, y_train_enc1)

# predictions on train set
y_pred_enc = pipe.predict(X_train_enc1)

# predictions on test set
y_pred_test_enc = pipe.predict(X_test_enc1)

pipe.score(X_test_enc1, y_test_enc1)

wmae_test(y_test_enc1, y_pred_test_enc)

df_encoded_new1 = df_encoded.copy()
df_encoded_new1.drop(drop_col, axis=1, inplace=True)

df_encoded_new1 = df_encoded_new1.drop(['Date'], axis=1)

df_encoded_new1 = df_encoded_new1.drop(['month'], axis=1)

# train-test split
train_data_enc_new1 = df_encoded_new1[:int(0.7*(len(df_encoded_new1)))]
test_data_enc_new1 = df_encoded_new1[int(0.7*(len(df_encoded_new1))):]

target = "Weekly_Sales"
used_cols3 = [c for c in df_encoded_new1.columns.to_list() if c not in [
    target]]  # all columns except price

X_train_enc2 = train_data_enc_new1[used_cols3]
X_test_enc2 = test_data_enc_new1[used_cols3]
y_train_enc2 = train_data_enc_new1[target]
y_test_enc2 = test_data_enc_new1[target]

# modeling part
pipe = make_pipeline(scaler, rf)

pipe.fit(X_train_enc2, y_train_enc2)

# predictions on train set
y_pred_enc = pipe.predict(X_train_enc2)

# predictions on test set
y_pred_test_enc = pipe.predict(X_test_enc2)

pipe.score(X_test_enc2, y_test_enc2)

wmae_test(y_test_enc2, y_pred_test_enc)

# result df for showing results together
df_results = pd.DataFrame(columns=["Model", "Info", 'WMAE'])

# writing results to df
df_results = df_results.append({
    "Model": 'RandomForestRegressor',
    "Info": 'w/out divided holiday columns',
    'WMAE': 5850}, ignore_index=True)


df_results = df_results.append({
    "Model": 'RandomForestRegressor',
    "Info": 'w/out month column',
    'WMAE': 5494}, ignore_index=True)
df_results = df_results.append({
    "Model": 'RandomForestRegressor',
    "Info": 'whole data',
    'WMAE': 2450}, ignore_index=True)
df_results = df_results.append({
    "Model": 'RandomForestRegressor',
    "Info": 'whole data with feature selection',
    'WMAE': 1801}, ignore_index=True)
df_results = df_results.append({
    "Model": 'RandomForestRegressor',
    "Info": 'whole data with feature selection w/out month',
    'WMAE': 2093}, ignore_index=True)

df_results

df.head()  # to see my data

# changing data to datetime for decomposing
df["Date"] = pd.to_datetime(df["Date"])

df.set_index('Date', inplace=True)  # seting date as index

plt.figure(figsize=(16, 6))
df['Weekly_Sales'].plot()
plt.show()

df_week = df.resample('W').mean()  # resample data as weekly

plt.figure(figsize=(16, 6))
df_week['Weekly_Sales'].plot()
plt.title('Average Sales - Weekly')
plt.show()

df_month = df.resample('MS').mean()  # resampling as monthly

plt.figure(figsize=(16, 6))
df_month['Weekly_Sales'].plot()
plt.title('Average Sales - Monthly')
plt.show()

# finding 2-weeks rolling mean and std
roll_mean = df_week['Weekly_Sales'].rolling(window=2, center=False).mean()
roll_std = df_week['Weekly_Sales'].rolling(window=2, center=False).std()

fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(df_week['Weekly_Sales'], color='blue', label='Average Weekly Sales')
ax.plot(roll_mean, color='red', label='Rolling 2-Week Mean')
ax.plot(roll_std, color='black', label='Rolling 2-Week Standard Deviation')
ax.legend()
fig.tight_layout()

adfuller(df_week['Weekly_Sales'])

train_data = df_week[:int(0.7*(len(df_week)))]
test_data = df_week[int(0.7*(len(df_week))):]

print('Train:', train_data.shape)
print('Test:', test_data.shape)

target = "Weekly_Sales"
used_cols = [c for c in df_week.columns.to_list() if c not in [
    target]]  # all columns except price

# assigning train-test X-y values

X_train = train_data[used_cols]
X_test = test_data[used_cols]
y_train = train_data[target]
y_test = test_data[target]

train_data['Weekly_Sales'].plot(
    figsize=(20, 8), title='Weekly_Sales', fontsize=14)
test_data['Weekly_Sales'].plot(
    figsize=(20, 8), title='Weekly_Sales', fontsize=14)
plt.show()

# decomposing of weekly data
decomposed = decompose(df_week['Weekly_Sales'].values, 'additive', m=20)

decomposed_plot(decomposed, figure_kwargs={'figsize': (16, 10)})
plt.show()

df_week_diff = df_week['Weekly_Sales'].diff(
).dropna()  # creating difference values

# taking mean and std of differenced data
diff_roll_mean = df_week_diff.rolling(window=2, center=False).mean()
diff_roll_std = df_week_diff.rolling(window=2, center=False).std()

fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(df_week_diff, color='blue', label='Difference')
ax.plot(diff_roll_mean, color='red', label='Rolling Mean')
ax.plot(diff_roll_std, color='black', label='Rolling Standard Deviation')
ax.legend()
fig.tight_layout()

df_week_lag = df_week['Weekly_Sales'].shift().dropna()  # shifting the data


lag_roll_mean = df_week_lag.rolling(window=2, center=False).mean()
lag_roll_std = df_week_lag.rolling(window=2, center=False).std()

fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(df_week_lag, color='blue', label='Difference')
ax.plot(lag_roll_mean, color='red', label='Rolling Mean')
ax.plot(lag_roll_std, color='black', label='Rolling Standard Deviation')
ax.legend()
fig.tight_layout()

logged_week = np.log1p(df_week['Weekly_Sales']).dropna()  # taking log of data

log_roll_mean = logged_week.rolling(window=2, center=False).mean()
log_roll_std = logged_week.rolling(window=2, center=False).std()

fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(logged_week, color='blue', label='Logged')
ax.plot(log_roll_mean, color='red', label='Rolling Mean')
ax.plot(log_roll_std, color='black', label='Rolling Standard Deviation')
ax.legend()
fig.tight_layout()

train_data_diff = df_week_diff[:int(0.7*(len(df_week_diff)))]
test_data_diff = df_week_diff[int(0.7*(len(df_week_diff))):]

# train_data = train_data['Weekly_Sales']
# test_data = test_data['Weekly_Sales']

model_auto_arima = auto_arima(train_data_diff, trace=True, start_p=0, start_q=0, start_P=0, start_Q=0,
                              max_p=20, max_q=20, max_P=20, max_Q=20, seasonal=True, maxiter=200,
                              information_criterion='aic', stepwise=False, suppress_warnings=True, D=1, max_D=10,
                              error_action='ignore', approximation=False)
model_auto_arima.fit(train_data_diff)

y_pred = model_auto_arima.predict(n_periods=len(test_data_diff))
y_pred = pd.DataFrame(y_pred, index=test_data.index, columns=['Prediction'])
plt.figure(figsize=(20, 6))
plt.title('Prediction of Weekly Sales Using Auto-ARIMA', fontsize=20)
plt.plot(train_data_diff, label='Train')
plt.plot(test_data_diff, label='Test')
plt.plot(y_pred, label='Prediction of ARIMA')
plt.legend(loc='best')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Weekly Sales', fontsize=14)
plt.show()

model_holt_winters = ExponentialSmoothing(train_data_diff, seasonal_periods=20, seasonal='additive',
                                          trend='additive', damped=True).fit()  # Taking additive trend and seasonality.
y_pred = model_holt_winters.forecast(
    len(test_data_diff))  # Predict the test data

# Visualize train, test and predicted data.
plt.figure(figsize=(20, 6))
plt.title('Prediction of Weekly Sales using ExponentialSmoothing', fontsize=20)
plt.plot(train_data_diff, label='Train')
plt.plot(test_data_diff, label='Test')
plt.plot(y_pred, label='Prediction using ExponentialSmoothing')
plt.legend(loc='best')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Weekly Sales', fontsize=14)
plt.show()


wmae_test(test_data_diff, y_pred)

am = arch_model(train_data_diff, vol='GARCH', power=1, o=1)
res = am.fit(update_freq=5)
print(res.summary())
