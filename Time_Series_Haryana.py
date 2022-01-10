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
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima

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

# Aggregating sales value over date
sf_new = sf_h.groupby('ACTUALDATE',as_index = False).sum()
sf_new.head()

# date column conversion and indexing
sf_new.info()
sf_new['ACTUALDATE'] = pd.to_datetime(sf_new['ACTUALDATE'] )
sf_new.set_index('ACTUALDATE',inplace=True)
sf_new.info()
#sorting the data
sf_1 = sf_new.sort_values('ACTUALDATE')
sf_1.head()
sf_1.tail()

#check central tendency attributes 
sf_1.describe()

#histogram
sns.distplot(sf_1.VALUE)

#boxplot
sf_1.boxplot()

# plot the data 
sf_1.plot(figsize = (15,6), title ='Haryana',color='purple')

sf_1.to_csv('Haryana.csv')


# Dicky fuller test
adfuller_test(sf_1)
'''

adfuller_test(sf_1)
Test statistics : -4.509315318242511
p-value : 0.00018915761236214482
#lags-used : 0
No of obs used : 45
'''

#lag plot
lag_plot(sf_1)

#auto correlation plot
autocorrelation_plot(sf_1,color='purple')

# time series decomposition
c= seasonal_decompose(sf_1,period=1 , model='mul')
c.plot()
c.observed
c.trend
c.seasonal
c.resid


#auto arima

stepwise_model = auto_arima(sf_1, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=False,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

'''
Best model:  ARIMA(0,1,1)(0,1,1)[12]          
Total fit time: 2.275 seconds
'''

stepwise_model.aic()
# Out[607]: 829.9356413523361

#model diagnostics
stepwise_model.plot_diagnostics(figsize=(7,5))

#train / test split

train = sf_1.loc['2014-01-01':'2016-12-01']
len(train) # 36
train.head()

test = sf_1.loc['2017-01-01':]
len(test)
#adding future dates to test data
from pandas.tseries.offsets import DateOffset

future_dates=[test.index[-1] + DateOffset(months=x) for x in range(0,24)]
future_dates
future_dates_test =pd.DataFrame(index=future_dates[1:])
future_test= pd.concat([test,future_dates_test],axis=1)
len(future_test)
test = future_test


stepwise_model.fit(train)
future_forecast = stepwise_model.predict(n_periods=10)
future_forecast

future_forecast = pd.DataFrame(future_forecast,index = test.index,columns=['Prediction'])
future_forecast.head()
future_test.head()
len(test)
len(future_forecast)

future_forecast.plot()
pd.concat([test,future_forecast],axis=1).plot()


####  building the model with Out of Cross Validation technique

model = ARIMA(train, order=(6,0,5))
model_fit = model.fit(disp=1)


### Fitted values
plt.plot(sf_1)
plt.plot(model_fit.fittedvalues, color = 'Red')


# Forecast
fc, se, conf = model_fit.forecast(33, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
#lower_series = pd.Series(conf[:, 0], index=test.index)
#upper_series = pd.Series(conf[:, 1], index=test.index)


# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
'''plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
'''
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()


model_fit.plot_predict(1,72)
