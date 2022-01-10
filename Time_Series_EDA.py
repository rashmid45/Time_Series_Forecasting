# set the directory
import os
os.chdir('C:\\Users\\shardul\\Desktop\\Rashmi\\Sales_Forecast')


#import libraries
import pandas as pd
pd.set_option('display.max_columns',None)
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

#import data
sf = pd.read_csv('Sales_Forecast_Data.csv')
sf1 = sf
sf1.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 230152 entries, 0 to 230151
Data columns (total 10 columns):
 #   Column      Non-Null Count   Dtype  
---  ------      --------------   -----  
 0   FIN_YEAR    230152 non-null  int64  
 1   NUMBER      230152 non-null  int64  
 2   DAY         230152 non-null  int64  
 3   MONTH       230152 non-null  object 
 4   ACTUALDATE  230152 non-null  object 
 5   STATE       230152 non-null  object 
 6   FG          230152 non-null  object 
 7   DISTRICT    230152 non-null  object 
 8   COMPANY     230152 non-null  object 
 9   VALUE       230152 non-null  float64
dtypes: float64(1), int64(3), object(6)
memory usage: 17.6+ MB
'''
sf1.columns
'''
Out[15]: 
Index(['FIN_YEAR', 'NUMBER', 'DAY', 'MONTH', 'ACTUALDATE', 'STATE', 'FG',
       'DISTRICT', 'COMPANY', 'VALUE'],
      dtype='object')
'''

for col in ['FIN_YEAR', 'MONTH', 'STATE', 'FG','DISTRICT', 'COMPANY']:
    sns.countplot(sf1[col])
    plt.title(col)
    plt.xticks(rotation=45)
    plt.show()

sf.shape
#Out[5]: (230152, 10)

sf = sf.drop(['FIN_YEAR','NUMBER','DAY','MONTH','FG','DISTRICT'], axis=1)
sf.head()
sf.shape
#Out[110]: (230152, 4)

# Filter only ABC manufacturing data
sf = sf[sf.COMPANY=='ABC Manufacturing']
sf.shape
# Out[112]: (28640, 4)

sf[sf.STATE=='Haryana'].shape 
# (6028, 4)
sf[sf.STATE=='Himachal Pradesh'].shape
# (3880, 4)
sf[sf.STATE=='Punjab'].shape
# (3668, 4)
sf[sf.STATE=='Uttar Pradesh'].shape
# (13024, 4)
sf[sf.STATE=='Uttarakhand'].shape
# (2040, 4)

sf.head()

def adfuller_test(sales):
    result=adfuller(sales)
    labels=['Test statistics','p-value','#lags-used','No of obs used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value))

### State Wise Dataset Preparation

## HARYANA

sf_haryana = sf[sf.STATE=='Haryana']
sf_haryana.shape # (6028,4)

sf_haryana.head()
sf_haryana.tail()

#check missing values
sf_haryana.isnull().sum() # no missing values

## dataset correction
sf_h =sf_haryana.drop(['STATE','COMPANY'],axis=1)
sf_h.head()
sf_h.to_csv('sf_h.csv')
type(sf_h)

sf_new = sf_h.groupby('ACTUALDATE',as_index = False).sum()
sf_new.head()


#check central tendency attributes 
sf_new.describe()

#histogram
sns.distplot(sf_new.VALUE)

#boxplot
sf_new.boxplot()

# date column conversion and indexing
sf_new.info()
sf_new['ACTUALDATE'] = pd.to_datetime(sf_new['ACTUALDATE'] )
sf_new.set_index('ACTUALDATE',inplace=True)

sf_new.head()
sf_new.plot()
sf_new.to_csv('Haryana.csv')


#sorting the data
sf_1 = sf_new.sort_index()
sf_1.head()
sf_1.tail()


# Dicky fuller test
adfuller_test(sf_1)



#lag plot
lag_plot(sf_1)

#auto correlation plot
autocorrelation_plot(sf_1)

# time series decomposition
c= seasonal_decompose(sf_1,period=1 , model='mul')
c.plot()
c.observed
c.trend
c.seasonal
c.resid
sf_1.to_csv('J2_h.csv')


from pandas import read_csv
from statsmodels.tsa.arima_model import ARIMA

series = pd.read_csv('J1.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
series.index = series.index.to_period('M')
series.head()
# fit model
model = ARIMA(series, order=(2,1,2))
model_fit = model.fit()
# summary of fit model
print(model_fit.summary())

# Actual vs Fitted
model_fit.plot_predict()

train = series[:80]
test = series[80:]


model = ARIMA(train, order=(1, 1, 1))  
fitted = model.fit(disp=-1)

fc, se, conf = fitted.forecast(12, alpha=0.05)  # 95% conf
test.head()

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

plt.plot(train, label='training')
plt.plot(test, label='actual')


# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()


#auto arima
from pmdarima.arima import auto_arima

stepwise_model = auto_arima(sf_1, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

'''
Performing stepwise search to minimize aic
 ARIMA(1,1,1)(0,1,1)[12]             : AIC=831.285, Time=3.42 sec
 ARIMA(0,1,0)(0,1,0)[12]             : AIC=851.614, Time=0.09 sec
 ARIMA(1,1,0)(1,1,0)[12]             : AIC=836.489, Time=0.53 sec
 ARIMA(0,1,1)(0,1,1)[12]             : AIC=829.936, Time=0.30 sec
 ARIMA(0,1,1)(0,1,0)[12]             : AIC=836.588, Time=0.13 sec
 ARIMA(0,1,1)(1,1,1)[12]             : AIC=831.654, Time=0.40 sec
 ARIMA(0,1,1)(0,1,2)[12]             : AIC=831.633, Time=0.98 sec
 ARIMA(0,1,1)(1,1,0)[12]             : AIC=830.897, Time=0.37 sec
 ARIMA(0,1,1)(1,1,2)[12]             : AIC=833.627, Time=1.70 sec
 ARIMA(0,1,0)(0,1,1)[12]             : AIC=839.662, Time=0.20 sec
 ARIMA(0,1,2)(0,1,1)[12]             : AIC=831.905, Time=0.74 sec
 ARIMA(1,1,0)(0,1,1)[12]             : AIC=836.629, Time=0.28 sec
 ARIMA(1,1,2)(0,1,1)[12]             : AIC=833.910, Time=1.03 sec
 ARIMA(0,1,1)(0,1,1)[12] intercept   : AIC=832.488, Time=0.46 sec

Best model:  ARIMA(0,1,1)(0,1,1)[12]          
Total fit time: 10.720 seconds
'''

stepwise_model.aic()
#Out[59]: 829.9356413523361

#model diagnostics
stepwise_model.plot_diagnostics(figsize=(7,5))

#train / test split

sf_1.info()
sf_1.head()
train = sf_1.loc['2014-01-01':'2016-12-01']
len(train) # 36
train.tail()
train.info()
sf_1.info()

test = sf_1.loc['2017-01-01':]
len(test)
test.tail()
test.info()

stepwise_model.fit(train)
future_forecast = stepwise_model.predict(n_periods=10)
future_forecast

sf_1.plot()
future_forecast.plot(color='darkgreen')

future_forecast = pd.DataFrame(future_forecast,index = test.index,columns=['Prediction'])
future_forecast
test.head()
len(test)
len(future_forecast)
pd.concat([test,future_forecast],axis=1).plot()

future_forecast2 = future_forcast

## HIMACHAL PRADESH

sf_HM = sf[sf.STATE=='Himachal Pradesh']
sf_HM.shape # (3880, 4)
sf_HM.head()
sf_HM.tail()
#check missing values
sf_HM.isnull().sum() # no missing values

## dataset correction
sf_HM_new =sf_HM.drop(['STATE','COMPANY'],axis=1)
sf_HM_new.head()

sf_HMnew = sf_HM_new.groupby(sf_HM_new.ACTUALDATE,as_index = False).sum()

sf_HMnew.head()

#check central tendency attributes 
sf_HMnew.describe()

#histogram
sns.distplot(sf_HMnew.VALUE)

#boxplot
sf_HMnew.boxplot()

# date column conversion and indexing
sf_HMnew.info()
sf_HMnew['ACTUALDATE'] = pd.to_datetime(sf_HMnew['ACTUALDATE'] )
sf_HMnew.set_index('ACTUALDATE',inplace=True)
sf_HMnew.head()
sf_HMnew.plot(figsize=(15,5))
sf_HMnew.to_csv('Himachal_Pradesh.csv')
 
# Dicky Fuller test
adfuller_test(sf_HMnew)

#Lag_plot
lag_plot(sf_HMnew)

# Auto correlation plot
autocorrelation_plot(sf_HMnew)

#time series decomposition
c= seasonal_decompose(sf_HMnew,period=1 , model='mul')

c.plot()
c.observed
c.trend
c.seasonal
c.resid

## PUNJAB

sf_punjab= sf[sf.STATE=='Punjab']
sf_punjab.shape (3668, 4)
sf_punjab.columns
sf_punjab.head()
sf_punjab.tail()
#check missing values
sf_punjab.isnull().sum() # no missing values

## dataset correction
sf_punjab =sf_punjab.drop(['STATE','COMPANY'],axis=1)
sf_punjab.head()

sf_PNew = sf_punjab.groupby(sf_punjab.ACTUALDATE,as_index = False).sum()
sf_PNew.head()

#check central tendency attributes 
sf_PNew.describe()

#histogram
sns.distplot(sf_PNew.VALUE)

#boxplot
sf_PNew.boxplot()

# date column conversion and indexing
sf_PNew.info()
sf_PNew['ACTUALDATE'] = pd.to_datetime(sf_PNew['ACTUALDATE'] )
sf_PNew.set_index('ACTUALDATE',inplace=True)
sf_PNew.head()
sf_PNew.plot(figsize=(15,5))
sf_PNew.to_csv('Punjab.csv')

# Dicky fuller test
adfuller_test(sf_PNew)

# Lag plot
lag_plot(sf_PNew)

# Auto correlation plot
autocorrelation_plot(sf_PNew)

# time series decomposition
c= seasonal_decompose(sf_PNew,period=1 , model='mul')

c.plot()
c.observed
c.trend
c.seasonal
c.resid


## UTTAR PRADESH

sf_UP= sf[sf.STATE=='Uttar Pradesh']
sf_UP.shape # (13024, 4)
sf_UP.columns
sf_UP.head()
sf_UP.tail()
#check missing values
sf_UP.isnull().sum() # no missing values

## dataset correction
sf_UP =sf_UP.drop(['STATE','COMPANY'],axis=1)
sf_UP.head()

sf_UPnew = sf_UP.groupby(sf_UP.ACTUALDATE,as_index = False).sum()
sf_UPnew.head()

#check central tendency attributes 
sf_UPnew.describe()

#histogram
sns.distplot(sf_UPnew.VALUE)

#boxplot
sf_UPnew.boxplot()

# date column conversion and indexing
sf_UPnew.info()
sf_UPnew['ACTUALDATE'] = pd.to_datetime(sf_UPnew['ACTUALDATE'] )
sf_UPnew.set_index('ACTUALDATE',inplace=True)
sf_UPnew.head()
sf_UPnew.plot(figsize=(15,5))
sf_UPnew.to_csv('Uttar_Pradesh.csv')

# Dicky fuller test
adfuller_test(sf_UPnew)

# Lag_plot
lag_plot(sf_UPnew)

# Autocorrelation plot
autocorrelation_plot(sf_UPnew)

c= seasonal_decompose(sf_UPnew,period=1 , model='mul')

c.plot()
c.observed
c.trend
c.seasonal
c.resid

## UTTARAKHAND

sf_UK= sf[sf.STATE=='Uttarakhand']
sf_UK.shape # 2040 ,4
sf_UK.columns
sf_UK.head()
sf_UK.tail()
#check missing values
sf_UK.isnull().sum() # no missing values

## dataset correction
sf_UK =sf_UK.drop(['STATE','COMPANY'],axis=1)
sf_UK.head()

sf_UKnew = sf_UK.groupby(sf_UK.ACTUALDATE,as_index = False).sum()

sf_UKnew.head()

#check central tendency attributes 
sf_UKnew.describe()

#histogram
sns.distplot(sf_UKnew.VALUE)

#boxplot

sf_UKnew.boxplot()

# date column conversion and indexing
sf_UKnew['ACTUALDATE'] = pd.to_datetime(sf_UKnew['ACTUALDATE'] )
sf_UKnew.set_index('ACTUALDATE',inplace=True)
sf_UKnew.head()
sf_UKnew.plot(figsize=(15,5))
sf_UKnew.to_csv('Uttarakhand.csv')

# Dicky fuller test
adfuller_test(sf_UKnew)

#lag plot
lag_plot(sf_UKnew)

# Auto correlation plot
autocorrelation_plot(sf_UKnew)

# time series decomposition
c= seasonal_decompose(sf_UKnew,period=1 , model='mul')
c.plot()
c.observed
c.trend
c.seasonal
c.resid