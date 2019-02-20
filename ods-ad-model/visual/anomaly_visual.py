#!/opt/anaconda3/bin/python3

import pandas as pd
import numpy as np
import itertools

from datetime import datetime, timedelta, date
from collections import defaultdict

import plotly.figure_factory as ff
from plotly.offline import init_notebook_mode, plot, iplot
from plotly.io import write_image

from showforecast import show_forecast


def show_table(X, anomaly_days, report_period):
    
    X.time = X.time.dt.strftime("%d-%m-%Y %H:%M:%S")
    
    report_date = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    report_period = '{0} - {1}'.format(report_period[0].strftime("%d-%m-%Y %H:%M:%S"), report_period[1].strftime("%d-%m-%Y %H:%M:%S"))
    
    report_info = 'Дата формирования отчета: {0}'.format(report_date)
    
    anomaly_table = ff.create_table(X, height_constant=50)
    title = 'Дата формирования отчета: {0} <br> Обработаны данные за период: {1} <br> Аномалии за последние {2} дня:'.format(report_date, report_period, anomaly_days)
    anomaly_table.layout.margin.update({'t':120})
    anomaly_table.layout.update({'title': title})
    anomaly_table.layout.titlefont.update({'size': 16})

    return anomaly_table


if __name__ == "__main__":
    
    df = pd.read_csv('../models/model_result/autoencoder_err.csv', ';', infer_datetime_format=True, parse_dates=['time'])
    df['is_anomaly'] = df.groupby(['host','consumer_group'])['error'].transform(lambda x: np.abs(x - x.mean()) > 3 * x.std())
    days = 3
    timeinterval = df.time.diff().mode()

    for group in df.consumer_group.unique():
        X = defaultdict(dict)
        y = defaultdict(dict)
        anomaly = defaultdict(dict)
        for host in df[df.consumer_group == group].host.unique():
            y[host] = {metric: df[(df.host == host) & (df.consumer_group == group)][metric].values for metric in df.drop(columns=['time', 'host', 'consumer_group', 'error', 'is_anomaly']).columns}
            X[host] = {metric: df[(df.host == host) & (df.consumer_group == group)].time.values.astype('M8[ms]').astype('O') for metric in df.drop(columns=['time', 'host', 'consumer_group', 'error', 'is_anomaly']).columns}
            anomaly[host] = df[(df.host == host) & (df.consumer_group == group) & (df.is_anomaly == True)].time.values.astype('M8[ms]').astype('O')
        fig_reqs = show_forecast(X, y, 'host ', '', group, anomaly, timeinterval, [df.time.max()-timedelta(days=days), df.time.max()])
        plot(fig_reqs, filename='../server/src/anomaly/report_{0}.html'.format(group), auto_open=False, show_link=False)

    
    report = show_table(df[(df.is_anomaly == True) & (df.time > df.time.max() - timedelta(days=days))][['time', 'consumer_group', 'host']].sort_values(by=['time'], ascending=False), days, [df.time.min(), df.time.max()])
    write_image(report, '../server/images/anomaly/anomaly_table.png')
        

