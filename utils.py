import os
import pandas as pd
from datetime import datetime
# from urllib.request import urlopen
# from io import BytesIO
# from zipfile import ZipFile


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def update_log(file_name):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S (UTC+07:00)")
    log = f'{file_name} downloaded at {current_time}\n'
    with open('download-log.txt', 'a') as f:
        f.write(log)
        f.close()

def download_to_csv(date, symbol, periodical='monthly'):
    url = f'https://data.binance.vision/data/spot/{periodical}/klines/{symbol}/1d/{symbol}-1d-{date}.zip'
    file_name = f'{symbol}-1d-{date}'
    df = pd.DataFrame()
    try:
        # http_response = urlopen(url)
        # zipfile = ZipFile(BytesIO(http_response.read()))
        # csv_file = zipfile.open(f'{file_name}.csv')
        cols = {1:'open', 2:'high', 3:'low', 4:'close', 5:'volume'}
        df = pd.read_csv(url, compression='zip',header=None, usecols=[1,2,3,4,5])
        df = df.rename(columns=cols)
        df['date'] = pd.date_range(start=date, periods=len(df))
    except:
        print(f'{file_name} unable to download.')
        
    return df