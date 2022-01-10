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

sf[sf.STATE=='Himachal Pradesh'].shape
# (3880, 4)


sf.head()

def adfuller_test(sales):
    result=adfuller(sales)
    labels=['Test statistics','p-value','#lags-used','No of obs used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value))

### State Wise Dataset Preparation

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

sf_HMnew = sf_HM_new.groupby('ACTUALDATE',as_index = False).sum()
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


#sorting the data
sf_5 = sf_HMnew.sort_values('ACTUALDATE')
sf_5.head()
sf_5.tail()

sf_5.plot(figsize=(15,6),color='Pink',title='Himachal Pradesh')
sf_5.to_csv('Himachal_Pradesh.csv')

# Dickey Fuller test
adfuller_test(sf_5)

#Lag_plot
lag_plot(sf_5)

# Auto correlation plot
autocorrelation_plot(sf_5,color='Pink')

#time series decomposition
c= seasonal_decompose(sf_5,period=1 , model='mul')

c.plot()
c.observed
c.trend
c.seasonal
c.resid

#auto arima

stepwise_model = auto_arima(sf_5, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

'''
Best model:  ARIMA(0,1,1)(1,1,0)[12] intercept
Total fit time: 24.946 seconds
'''

stepwise_model.aic()
# Out[755]: 690.013374133266

#model diagnostics
stepwise_model.plot_diagnostics(figsize=(7,5))

#train / test split

sf_5.info()
sf_5.head()
train = sf_5.loc['2014-01-01':'2016-12-01']
len(train) # 36

test = sf_5.loc['2017-01-01':]
len(test)


stepwise_model.fit(train)
future_forecast = stepwise_model.predict(n_periods=10)
future_forecast

future_forecast = pd.DataFrame(future_forecast,index = test.index,columns=['Prediction'])
future_forecast
test
len(test)
len(future_forecast)
pd.concat([test,future_forecast],axis=1).plot()




