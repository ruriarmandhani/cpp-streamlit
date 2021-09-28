import pandas as pd
import calendar
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial
from utils import download_to_csv, update_log

def download(symbol:str, start_date:str, end_date:str):
    year_end = int(end_date.split('-')[0])
    month_end = int(end_date.split('-')[1])
    last_day = calendar.monthrange(year_end, month_end)[1]
    date_range = pd.date_range(start=start_date, end=f'{end_date}-{last_day}', freq='M')
    date_range = date_range.format(formatter=lambda x: x.strftime('%Y-%m'))
    
    pool = Pool(processes=cpu_count()-1)
    with pool as p:
        df = list(p.map(partial(download_to_csv, symbol=symbol), date_range))
        df = pd.concat(df)

        if datetime.now() <= pd.to_datetime(f'{end_date}-{last_day}'):
            date_range = pd.date_range(start=end_date, end=datetime.now(), freq='D')
            date_range = date_range.format(formatter=lambda x: x.strftime('%Y-%m-%d'))
            df_daily = list(p.map(partial(download_to_csv, symbol=symbol, periodical='daily'), date_range))
            df_daily = pd.concat(df_daily)
            df = pd.concat([df, df_daily])
            # df = df.reset_index(drop=True)
            
    min_date = df['date'].min().strftime('%Y-%m-%d')
    max_date = df['date'].max().strftime('%Y-%m-%d')
    df.to_csv(f'./binance/{symbol}-price.csv', index=False)
    update_log(f'{symbol}-1d-{min_date}-to-{max_date}')