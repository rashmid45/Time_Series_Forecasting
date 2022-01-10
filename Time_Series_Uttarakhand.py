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


sf[sf.STATE=='Uttarakhand'].shape
# (2040, 4)

sf.head()

def adfuller_test(sales):
    result=adfuller(sales)
    labels=['Test statistics','p-value','#lags-used','No of obs used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value))

### State Wise Dataset Preparation

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

sf_UKNew = sf_UK.groupby(sf_UK.ACTUALDATE,as_index = False).sum()
sf_UKNew.head()

#check central tendency attributes 
sf_UKNew.describe()

#histogram
sns.distplot(sf_UKNew.VALUE)

#boxplot
sf_UKNew.boxplot()

# date column conversion and indexing
sf_UKNew['ACTUALDATE'] = pd.to_datetime(sf_UKNew['ACTUALDATE'] )
sf_UKNew.set_index('ACTUALDATE',inplace=True)
sf_UKNew.head()

#sorting the data
sf_3 = sf_UKNew.sort_values('ACTUALDATE')
sf_3.head()
sf_3.tail()

sf_3.plot(figsize=(15,6),color='turquoise')
sf_3.to_csv('Uttarakhand.csv')

# Dicky fuller test
adfuller_test(sf_3)

#lag plot
lag_plot(sf_3)

# Auto correlation plot
autocorrelation_plot(sf_3,color='turquoise')

# time series decomposition
c= seasonal_decompose(sf_3,period=1 , model='mul')
c.plot()
c.observed
c.trend
c.seasonal
c.resid


#auto arima
stepwise_model = auto_arima(sf_3, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

'''
Best model:  ARIMA(3,1,0)(0,1,1)[12]          
Total fit time: 23.346 seconds
'''

stepwise_model.aic()
# Out[352]: 745.7260256864411

#model diagnostics
stepwise_model.plot_diagnostics(figsize=(7,5))

#train / test split

sf_3.info()
sf_3.head()
train = sf_3.loc['2014-01-01':'2016-12-01']
len(train) # 36

test = sf_3.loc['2017-01-01':]

stepwise_model.fit(train)
future_forecast = stepwise_model.predict(n_periods=10)
future_forecast

future_forecast = pd.DataFrame(future_forecast,index = test.index,columns=['Prediction'])
future_forecast
test.head()
len(test)
len(future_forecast)
pd.concat([test,future_forecast],axis=1).plot()

future_forecast2 = future_forcast
