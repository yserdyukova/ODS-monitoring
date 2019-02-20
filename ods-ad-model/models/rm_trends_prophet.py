#!/opt/anaconda3/bin/python3

import pandas as pd
import numpy as np
import itertools
from collections import defaultdict
from datetime import datetime, timedelta, date
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.diagnostics import cross_validation

def prophet(df, holidays):
    ''' Function for prophet calculation
    '''

    sample = df.copy(deep=True)
    sample.columns = ['ds','y']

    
    modelProphet = Prophet(changepoint_prior_scale=0.1,
                           yearly_seasonality=False,
                           weekly_seasonality=True,
                           holidays=holidays)
    modelProphet.fit(sample)

    future = modelProphet.make_future_dataframe(periods=0, freq='20min')
    
    return modelProphet.predict(future).trend


def main(inputfile, outputfile, mode='group'):
    
    holidays = {
    2018: {1:[1,2,3,4,5,6,7,8,13,14,20,21,27,28],2:[3,4,10,11,17,18,22,23,24,25],3:[3,4,7,8,9,10,11,17,18,24,25,31],4:[1,7,8,14,15,21,22,28,29,30],5:[1,2,5,6,8,9,12,13,19,20,26,27],6:[2,3,9,10,11,12,16,17,23,24,30],7:[1,7,8,14,15,21,22,28,29],8:[4,5,11,12,18,19,25,26],9:[1,2,8,9,15,16,22,23,29,30],10:[6,7,13,14,20,21,27,28],11:[3,4,5,10,11,17,18,24,25],12:[1,2,8,9,15,16,22,23,29,30,31]},
    2019: {1:[1,2,3,4,5,6,7,8,12,13,19,20,26,27],2:[2,3,9,10,16,17,22,23,24],3:[2,3,7,8,9,10,16,17,23,24,30,31],4:[6,7,13,14,20,21,27,28,30],5:[1,2,3,4,5,8,9,10,11,12,18,19,25,26],6:[1,2,8,9,11,12,15,16,22,23,29,30],7:[6,7,13,14,20,21,27,28],8:[3,4,10,11,17,18,24,25,31],9:[1,7,8,14,15,21,22,28,29],10:[5,6,12,13,19,20,26,27],11:[2,3,4,9,10,16,17,23,24,30],12:[1,7,8,14,15,21,22,28,29,31]},
    2020: {1:[1,2,3,4,5,6,7,8,11,12,18,19,25,26],2:[1,2,8,9,15,16,22,23,24,29],3:[1,7,8,9,14,15,21,22,28,29],4:[4,5,11,12,18,19,25,26,30],5:[1,2,3,8,9,10,11,16,17,23,24,30,31],6:[6,7,11,12,13,14,20,21,27,28],7:[4,5,11,12,18,19,25,26],8:[1,2,8,9,15,16,22,23,29,30],9:[5,6,12,13,19,20,26,27],10:[3,4,10,11,17,18,24,25,31],11:[1,3,4,7,8,14,15,21,22,28,29],12:[5,6,12,13,19,20,26,27,31]},
    2021: {1:[1,2,3,4,5,6,7,8,9,10,16,17,23,24,30,31],2:[6,7,13,14,20,21,22,23,27,28],3:[6,7,8,13,14,20,21,27,28],4:[3,4,10,11,17,18,24,25,30],5:[1,2,3,8,9,10,15,16,22,23,29,30],6:[5,6,11,12,13,14,19,20,26,27],7:[3,4,10,11,17,18,24,25,31],8:[1,7,8,14,15,21,22,28,29],9:[4,5,11,12,18,19,25,26],10:[2,3,9,10,16,17,23,24,30,31],11:[3,4,6,7,13,14,20,21,27,28],12:[4,5,11,12,18,19,25,26,31]}
    }
    holidays = pd.DataFrame(list(itertools.chain.from_iterable(list(itertools.chain.from_iterable([[[date(year,month,day) for day in days] for month,days in v.items()] for year,v in holidays.items()])))),columns=['ds'])
    holidays['holiday'] = 'holiday'
    
    
    df = pd.read_csv(inputfile, ';', infer_datetime_format=True, parse_dates=['time'])
    
    ind = ['host'] if mode == 'sys' else  ['host', 'consumer_group']
    df = df.set_index(ind)
    
    forecast_prophet = pd.DataFrame()
    for metric in df.drop(columns=['time']).columns:
        forecast_key = pd.DataFrame()
        for key in df.index.unique().values:
            sample = df.loc[key, ('time', metric)].reset_index()
            sample[metric] = prophet(sample[['time', metric]], holidays)
            forecast_key = pd.concat([forecast_key, sample], sort=False)
            
        forecast_prophet = pd.concat([forecast_prophet, forecast_key.set_index(['time', *ind])], axis=1, sort=False)
        
    forecast_prophet.reset_index().to_csv(outputfile, ';', index=False)
    

if __name__ == "__main__":
    
    main('../features/clear_data/rm_features_for_trends_by_metrics.csv', 'model_result/prophet_trends_by_metrics.csv', 'group')
    main('../features/clear_data/rm_features_for_trends_by_sysmetrics.csv', 'model_result/prophet_trends_by_sysmetrics.csv', 'sys')

