#!/opt/anaconda3/bin/python3

import pandas as pd
import numpy as np
from plotly.offline import init_notebook_mode, plot, iplot
from collections import defaultdict
from showforecast import show_forecast

if __name__ == "__main__":
    
    
    df = pd.read_csv('../models/model_result/prophet_trends_by_metrics.csv', ';', infer_datetime_format=True, parse_dates=['time'])
    timeinterval = df.time.diff().mode()
    for metric in df.drop(columns=['time', 'host', 'consumer_group']).columns:   
        X = defaultdict(dict)
        y = defaultdict(dict)
        for host in df.host.unique():
            y[host] = {group: df[(df.host == host) & (df.consumer_group == group)][metric].values for group in df[df.host == host].consumer_group.unique()}
            X[host] = {group: df[(df.host == host) & (df.consumer_group == group)].time.values.astype('M8[ms]').astype('O') for group in df[df.host == host].consumer_group.unique()}
        fig_reqs = show_forecast(X, y, 'host ', '', metric, timeinterval=timeinterval)
        plot(fig_reqs, filename='../server/src/trends/by_metrics/trend_{0}.html'.format(metric), auto_open=False, show_link=False)

        
    df = pd.read_csv('../models/model_result/prophet_trends_by_sysmetrics.csv', ';', infer_datetime_format=True, parse_dates=['time'])
    timeinterval = df.time.diff().mode()
    X = defaultdict(dict)
    y = defaultdict(dict)
    for metric in df.drop(columns=['time', 'host']).columns:   
        y[metric] = {host: df[(df.host == host)][metric].values for host in df.host.unique()}
        X[metric] = {host: df[(df.host == host)].time.values.astype('M8[ms]').astype('O') for host in df.host.unique()}
    fig_reqs = show_forecast(X, y, '', 'host ', 'sys metrics', timeinterval=timeinterval)    
    plot(fig_reqs, filename='../server/src/trends/sys_metrics/trend_sys_metrics.html', auto_open=False, show_link=False)
    

    df = pd.read_csv('../features/clear_data/rm_features_for_trends_by_metrics.csv', ';', infer_datetime_format=True, parse_dates=['time'])
    timeinterval = df.time.diff().mode()
    for metric in df.drop(columns=['time', 'host', 'consumer_group']).columns:   
        tmp_outliers = df.groupby(['host','consumer_group'])[metric].transform(lambda x: np.abs(x - x.mean()) > 5 * x.std())
        X = defaultdict(dict)
        y = defaultdict(dict)
        for host in df.host.unique():
            y[host] = {group: df[(df.host == host) & (df.consumer_group == group) & (tmp_outliers == False)][metric].values for group in df[df.host == host].consumer_group.unique()}
            X[host] = {group: df[(df.host == host) & (df.consumer_group == group) & (tmp_outliers == False)].time.values.astype('M8[ms]').astype('O') for group in df[df.host == host].consumer_group.unique()}
        fig_reqs = show_forecast(X, y, 'host ', '', metric, timeinterval=timeinterval)
        plot(fig_reqs, filename='../server/src/real/by_metrics/real_{0}.html'.format(metric), auto_open=False, show_link=False)

