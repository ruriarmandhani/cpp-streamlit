import json
import os
import pandas as pd
import plotter
import re
import streamlit as st
import subprocess
import sys
import tensorflow as tf
import pickle

from contextlib import contextmanager
from datetime import datetime, timedelta
from io import StringIO
from matplotlib import pyplot as plt
from model import preprocessing, train_model, eval_model, forecast
from PIL import Image
from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME
from threading import current_thread
from urllib.request import urlopen


@contextmanager
def st_redirect(src, dst):
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)
    
    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if getattr(current_thread(), REPORT_CONTEXT_ATTR_NAME, None):
                buffer.write(b)
                output_func(buffer.getvalue())
            else:
                old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write


@contextmanager
def st_stdout(dst):
    with st_redirect(sys.stdout, dst):
        yield


def retrain_model(df, params):
    try:
        st.markdown(
            """
            <style>
                pre {max-height:300px;overflow:auto;} 
                .stCodeBlock div {display:none;}
            <style/>""",
            unsafe_allow_html=True)
        with st_stdout('code'):
            df_train = preprocessing(df, params['lags'], params['col'])
            train_model(df_train, params)
            eval_model(df_train, params)
        st.success("The model has been successfully trained.")
    except:
        st.error("Failed to train the model. Try again.")


def read_data_monthly(df, date):
    mask = (df['date'].dt.strftime('%Y-%m') == date)
    df = df.loc[mask]
    df = df.reset_index(drop=True)
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    return df


# @st.cache(allow_output_mutation=True, suppress_st_warning=True)
def read_all_time_data(symbol):
    df = pd.read_csv(f'./binance/{symbol}-price.csv')
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    return df


@st.cache
def get_all_symbols():
    url = f'https://api.binance.com/api/v3/exchangeInfo'
    http_response = urlopen(url).read()
    json_response = json.loads(http_response)
    symbols = list(
        map(lambda symbol: symbol['symbol'], json_response['symbols']))
    return symbols


def read_csv(df):
    csv = df.to_csv(index=False).encode('utf-8')
    return csv


def get_tf_model(symbol, col):
    try:
        model = tf.keras.models.load_model(f'./model/{symbol}_model_{col}.h5')
    except:
        model = None
    return model


def main():
    st.set_page_config(page_title='Ruri Armandhani • Streamlit')

    st.markdown(""" <style> #MainMenu {visibility: hidden;}</style> """,
        unsafe_allow_html=True
    )

    header = st.container()
    radio = st.sidebar.radio('Navigation', ['Home', 'Data', 'Model', 'Price Prediction'])
    symbols = sorted(get_all_symbols())
    default_symbol = symbols.index('BTCUSDT')

    if radio != "Home":
        symbol = st.sidebar.selectbox(
            'Select Cryptocurrency Symbol:', symbols, index=default_symbol)

    warn_message = False
    
    try:
        df = read_all_time_data(symbol)
        start_date = df['date'].min()
        end_date = df['date'].max()
        cols = df.columns.tolist()
    except:
        df = pd.DataFrame()
        start_date = datetime.now()
        end_date = datetime.now()
        cols = []
        warn_message = True

    periods = (end_date.year - start_date.year) * 12 + \
        (end_date.month - start_date.month) + 1
    date_range = pd.date_range(start=start_date, periods=periods, freq='M')
    date_range = date_range.strftime('%Y-%m')

    if radio == "Home":
        st.markdown(
            """
            <style>
            #made-by-ruri-armandhani {text-align:center;}
            #cryptocurrency-price-prediction {text-align:center;}
            <style/>
            """, unsafe_allow_html=True)

        with header:
            st.title('Cryptocurrency Price Prediction')

        st.subheader('Made By: [Ruri Armandhani](https://www.linkedin.com/in/ruri-armandhani/)')
        st.markdown("---")
        st.markdown(
            """
            <style>.stMarkdown p{margin-bottom:0.3rem;}<style/>
            """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader('📌 Current Features:')
            st.markdown(
                """ 
                * 📂 Generate cryptocurrency price data.\n
                * 📊 Data visualization.\n
                * 📈 Predict cryptocurrency price with neural networks model.\n
                * ⚙️ Model training and tuning.\n
                """)
        with col2:
            st.subheader('💪 Features On Progress:')
            st.markdown(
                """ 
                * ✏️ Custom neural networks model.
                * 📋 Add method options for model validation.
                * 🛠️ Add options for feature engineering.
                """)
        st.markdown('---')
        st.subheader('Resources:')
        st.markdown(
            """
            * Data source:\
                [Binance Data Collection](https://data.binance.vision/),
                [Binance Public Data Documentation](https://github.com/binance/binance-public-data/)
            * Cryptocurrency symbols: [https://api.binance.com/api/v3/exchangeInfo]
                (https://api.binance.com/api/v3/exchangeInfo)
            * Web app framework: [Streamlit](https://docs.streamlit.io/)
            * Neural networks model: [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/)
            * Data manipulation and preprocessing:
                [Pandas](https://pandas.pydata.org/docs/reference/index.html),
                [NumPy](https://numpy.org/doc/stable/reference/index.html),
                [Scikit-Learn](https://scikit-learn.org/stable/modules/classes.html)
            * Data visualization: [Seaborn](https://seaborn.pydata.org/api.html),
                [Matplotlib](https://matplotlib.org/stable/contents.html)
            * Source code: [https://github.com/ruriarmandhani](https://github.com/ruriarmandhani)
            """, unsafe_allow_html=True)

    elif radio == 'Data':
        page = st.sidebar.selectbox(
            'Page Options', ['Data Generator', 'DataFrame', 'Data Visualization'])

        if page == 'Data Generator':
            with header:
                st.title('Data Generator')

            status = st.empty()
            if warn_message == True:
                status.warning(f'{symbol} dataframe does not exist and \
                    needs to be generated.')

            # gen_start_date = st.date_input('Start Date:', value=start_date, max_value=datetime.now())
            # gen_start_date = gen_start_date.strftime('%Y-%m')
            gen_start_date = st.text_input('Start Date (YYYY-MM):',
                value=start_date.strftime('%Y-%m'), max_chars=7)
            if not re.match("\d{4}-\d{2}$", gen_start_date):
                st.error('Input format for start date must be YYYY-MM.')

            # gen_end_date = st.date_input('End Date (YYYY-MM):', max_value=datetime.now())
            # gen_end_date = gen_end_date.strftime('%Y-%m')
            gen_end_date = st.text_input('End Date (YYYY-MM):',
                value=datetime.now().strftime('%Y-%m'), max_chars=7)
            if not re.match("\d{4}-\d{2}$", gen_end_date):
                st.error('Input format for end date must be YYYY-MM.')

            col1, col2 = st.columns([2.61, 1])
            with col1:
                button = st.button('Generate DataFrame')

            if button:
                try:
                    print('Updating DataFrame...')
                    subprocess.run(['python', 'downloader.py', symbol, 
                        gen_start_date, gen_end_date], shell=True)
                    print('Done.')
                    status.empty()
                    st.success(f'{symbol} dataframe has been successfully generated.')
                except:
                    st.error('Failed to generate dataframe. Try again.')

        elif page == 'DataFrame':
            with header:
                st.title('DataFrame')
            
            st.subheader(f'{symbol} Price Monthly')

            col1, col2 = st.columns([1, 1.5])
            with col1:
                date = st.selectbox('', sorted(date_range, reverse=True))
            with col2:
                if warn_message == False:
                    st.markdown(
                        """
                        <style>
                        .stDownloadButton {text-align:right; vertical-align:middle;}
                        .stSelectbox label {display:none;}
                        <style/>
                        """, unsafe_allow_html=True)

                    st.download_button('Download DataFrame', read_csv(df), 
                        f'{symbol}-price.csv', 'text/csv')
            
            try:
                df_monthly = read_data_monthly(df, date)
            except:
                df_monthly = pd.DataFrame()

            st.table(df_monthly)

            if warn_message == True:
                st.error(f'Unable to read {symbol} dataframe.\
                    You need to generate it first.')
                
        else:
            with header:
                st.title('Data Visualization')

            plots = ['Line Plot', 'Pair Plot', 'Distribution Plot',
                'Probability Plot']

            plot = st.sidebar.selectbox('Plot Options', plots, index=0)

            col1, col2 = st.columns([1, 1])
            with col1:
                start = st.date_input('Start Date:', value=start_date, 
                    min_value=start_date, max_value=end_date)
            with col2:
                end = st.date_input('End Date:', value=end_date,
                    min_value=start, max_value=end_date)
            
            if not df.empty:
                mask = (df['date'] >= pd.to_datetime(start)) & (
                    df['date'] <= pd.to_datetime(end))
                df_for_chart = df.loc[mask]
                
                with st.expander('Descriptive Statistics'):
                    with st_stdout('code'):
                        print(df.describe())

                if plot == 'Line Plot':
                    st.subheader('Line Plot')
                    cols_lp = st.multiselect('Choose column:', cols[:5], default=cols[3])
                    st.line_chart(data=df_for_chart.set_index('date')[cols_lp])
                elif plot == 'Pair Plot':
                    st.subheader('Pair Plot')
                    pair_plot = plotter.pair_plot(df_for_chart)
                    st.pyplot(pair_plot)
                elif plot == 'Distribution Plot':
                    st.subheader('Distribution Plot')
                    cfc_dist = st.selectbox('Choose column:', cols[:5], index=3, key='dist')
                    dist_plot = plotter.dist_plot(df_for_chart[[cfc_dist,'date']])
                    st.pyplot(dist_plot)
                else:
                    st.subheader('Probability Plot')
                    cfc_prob = st.selectbox('Choose column:', cols[:5], index=3, key='prob')
                    prob_plot = plotter.prob_plot(df_for_chart[cfc_prob])
                    plt.title('')
                    st.pyplot(prob_plot)
            else:
                st.error(
                    'Unable to visualize data because dataframe does not exist.')
    elif radio == 'Model':
        page = st.sidebar.selectbox('Page Options', ['Training', 'Evaluation'])

        if page == 'Training':
            with header:
                st.title('Model Training')

            if warn_message == True:
                st.warning(f'{symbol} dataframe does not exist.\
                    You need to generate it first before training the model.')

            col = st.selectbox(
                'Type of price:', cols[:4], index=3)

            if os.path.isfile(f'./model/{symbol}_model_{col}.h5'):
                st.info(f'This app already has a default model for \
                    {symbol} {col} price. You can also retrain the \
                    model by changing the hyperparameters and clicking\
                    the \'Train Model\' button below.')

            with st.form('hp_form'):
                st.subheader('Training-Validation Data Split')
                val_size = st.slider('Percentage for validation dataset:',
                                     min_value=0.1, max_value=0.5, value=0.3, step=0.1, format=f'%.1f')

                lags = st.slider(
                    'Lag Features:', min_value=1, max_value=30, value=7)

                st.subheader('Hyperparameters Tuning')

                epochs = st.slider('Epochs:', min_value=1,
                                   max_value=50, value=10)

                batch_size = st.slider(
                    'Batch Size:', min_value=2, max_value=64, value=32)

                loss_metric_dict = {
                    'Mean Squared Error': 'mse',
                    'Root Mean Squared Error': 'rmse',
                    'Mean Absolute Error': 'mae',
                    'Mean Absolute Percentage Error': 'mape'
                }

                col1, col2 = st.columns([1, 1])
                with col1:
                    lr = st.number_input('Learning Rate:', min_value=0.00001,
                                         max_value=1.0, value=0.001, step=0.00001, format="%.5f")
                    loss = st.selectbox('Loss:', list(loss_metric_dict.keys()))
                    loss = loss_metric_dict[loss]
                with col2:
                    optimizer = st.selectbox('Optimizer:',
                        ['Adam', 'SGD', 'RMSProp'])
                    metric = st.selectbox('Metric:', 
                        list(loss_metric_dict.keys()),
                        index=2)
                    metric = [loss_metric_dict[metric]]

                button = st.form_submit_button('Train Model')

            if button:
                params = {
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'learning_rate': lr,
                    'optimizer': optimizer,
                    'loss': loss,
                    'metric': metric,
                    'val_size': val_size,
                    'col': col,
                    'lags': lags,
                    'symbol': symbol
                }
                retrain_model(df, params)

        elif page == 'Evaluation':
            with header:
                st.title('Model Evaluation')
            col = st.sidebar.selectbox('Type of model:', cols[:4], index=3)
            
            try:
                with st.expander('Feature Engineering'):
                    msg = """
                        The model on this app only using lag features.
                        If you are not familiar with lag features,
                        I suggest you to read this 
                        [article](https://machinelearningmastery.com/basic-feature-engineering-time-series-data-python/).
                    """
                    st.write(msg)

                with st.expander('Architecture'):
                    _, col1 = st.columns([0.3,1])
                    with col1:
                        image = Image.open(f'./model/{symbol}_model_architecture.png')
                        st.image(image, caption='ANN Model')
                
                with st.expander('Training-Validation Error'):
                    image = Image.open(f'./graph/{symbol}_model_{col}_loss.png')
                    st.image(image, caption='Training and Validation Error',)
                
                with st.expander('Performance'):
                    image = Image.open(f'./graph/{symbol}_model_{col}_val.png')
                    st.image(image, caption='Real vs Prediction')

            except:
                st.error(
                    'Model does not exist. You need to train the model first.')

    else:
        with header:
            st.title('Price Prediction')
        col = st.sidebar.selectbox('Type of price:', cols[:4], index=3)
        
        model = get_tf_model(symbol, col)

        if model is None:
            st.error(f'The model does not exist. You need to train the model for {symbol} {col} price.')
        else:
            last_updated_date = df.iloc[-1]['date'].strftime('%Y-%m-%d')
            st.info(f'Last updated date for dataset: {last_updated_date}')
            days = st.slider('The number of days ahead:', min_value=1, max_value=30)
            pred_values = forecast(model, df, days)
            forecast_start_date = df.iloc[-1]['date'] + timedelta(days=1)
            forecast_end_date = forecast_start_date + timedelta(days=days-1)
            dates = pd.date_range(start=forecast_start_date, 
                end=forecast_end_date, freq='D')
            pred_df = pd.DataFrame({'date':dates, f'{col}_price_pred':pred_values})
            # pred_df['date'] = pred_df['date'].dt.strftime('%Y-%m-%d')

            with st.expander('Table of Results', True):
                st.table(pred_df)
            with st.expander('Graph', True):
                st.line_chart(pred_df.set_index('date'))

    # fixing line chart tooltip not working on fullscreen
    st.markdown(
        '<style>#vg-tooltip-element{z-index: 1000051}</style>', 
        unsafe_allow_html=True)           

if __name__ == '__main__':
    main()
