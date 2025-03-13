import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from prophet.plot import plot_plotly
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

def main():
    st.header("📈 Prediksi Stok", divider="green")
    conn = sqlite3.connect('data/stok.db')
    c = conn.cursor()

    # Ambil data produk
    produk = c.execute("SELECT id, nama FROM produk").fetchall()
    if not produk:
        st.warning("Tidak ada produk tersedia. Silakan tambah produk terlebih dahulu.", icon="⚠️")
        return

    # Parameter Prediksi
    with st.container():
        st.subheader("🛠️ Parameter Prediksi")
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        with col1:
            daftar_produk = [p[1] for p in produk]
            produk_pilihan = st.selectbox(
                "Produk yang akan diprediksi",
                daftar_produk,
                format_func=lambda x: f"📦 {x}",
                help="Pilih produk untuk prediksi"
            )
            produk_id = next((p[0] for p in produk if p[1] == produk_pilihan), None)
        with col2:
            periode_prediksi = st.slider(
                "Periode Prediksi (bulan)",
                1, 24, 12,
                help="Jumlah bulan yang akan diprediksi"
            )
        with col3:
            model_confidence = st.slider(
                "Tingkat Kepercayaan (%)",
                80, 99, 95,
                help="Interval kepercayaan prediksi"
            )
        with col4:
            model_choice = st.selectbox(
                "Metode Prediksi",
                ["ARIMA", "SARIMA", "Prophet", "LSTM"],
                help="Pilih algoritma prediksi"
            )

    # Ambil data historis
    query = """
    SELECT 
        strftime('%Y-%m', tk.tanggal) as bulan,
        SUM(tk.jumlah) as total_keluar
    FROM transaksi_keluar tk
    WHERE tk.produk_id = ?
    GROUP BY strftime('%Y-%m', tk.tanggal)
    ORDER BY strftime('%Y-%m', tk.tanggal)
    """
    df_history = pd.read_sql_query(query, conn, params=(produk_id,))

    if len(df_history) < 12 and model_choice in ["SARIMA", "Prophet", "LSTM"]:
        st.error("Data tidak cukup! Minimal 12 bulan data diperlukan untuk model musiman/Prophet/LSTM")
        return

    # Proses data historis
    df_history['bulan'] = pd.to_datetime(df_history['bulan'] + '-01')
    df_history.set_index('bulan', inplace=True)
    date_range = pd.date_range(start=df_history.index.min(), end=df_history.index.max(), freq='MS')
    df_history = df_history.reindex(date_range, fill_value=0)

    # Pemodelan
    if model_choice == "ARIMA":
        try:
            model = ARIMA(df_history['total_keluar'], order=(1,1,1))
            results = model.fit()
            forecast = results.get_forecast(steps=periode_prediksi)
            forecast_mean = forecast.predicted_mean
            conf_int = forecast.conf_int(alpha=(100 - model_confidence)/100)
        except Exception as e:
            st.error(f"Gagal membuat model ARIMA: {str(e)}")
            return

    elif model_choice == "SARIMA":
    try:
        model = SARIMAX(df_history['total_keluar'], 
                       order=(1,1,1), 
                       seasonal_order=(1,1,1,12))
        results = model.fit(disp=False)
        forecast = results.get_forecast(steps=periode_prediksi)
        forecast_mean = forecast.predicted_mean
        conf_int = forecast.conf_int(alpha=(100 - model_confidence)/100)
        
        # Perbaikan: Gunakan nama kolom yang valid
        df_forecast = pd.DataFrame({
            'bulan': future_dates,
            'prediksi': forecast_mean,
            'lower_ci': conf_int['lower'],  # Ganti indeks dengan nama kolom
            'upper_ci': conf_int['upper']   # Ganti indeks dengan nama kolom
        }).set_index('bulan')
    except Exception as e:
        st.error(f"Gagal membuat model SARIMA: {str(e)}")
        return

    elif model_choice == "Prophet":
        try:
            df_prophet = df_history.reset_index().rename(columns={'bulan':'ds', 'total_keluar':'y'})
            model = Prophet(interval_width=model_confidence/100)
            model.fit(df_prophet)
            future = model.make_future_dataframe(periods=periode_prediksi, freq='MS')
            forecast = model.predict(future)
            forecast_mean = forecast['yhat'][-periode_prediksi:].values
            conf_int = forecast[['yhat_lower', 'yhat_upper']][-periode_prediksi:].values.T
        except Exception as e:
            st.error(f"Gagal membuat model Prophet: {str(e)}")
            return

    elif model_choice == "LSTM":
        try:
            # Preprocessing data
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df_history.values)
            
            # Membuat dataset time series
            def create_dataset(data, time_step=1):
                X, y = [], []
                for i in range(len(data)-time_step-1):
                    X.append(data[i:(i+time_step), 0])
                    y.append(data[i + time_step, 0])
                return np.array(X), np.array(y)
            
            time_step = 12
            X, y = create_dataset(scaled_data, time_step)
            X = X.reshape(X.shape[0], X.shape[1], 1)
            
            # Membuat model LSTM
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
            model.add(LSTM(50, return_sequences=False))
            model.add(Dense(25))
            model.add(Dense(1))
            
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X, y, batch_size=1, epochs=20, verbose=0)
            
            # Prediksi
            test_input = scaled_data[-time_step:]
            test_input = test_input.reshape((1, time_step, 1))
            forecast_scaled = model.predict(test_input)
            forecast_mean = scaler.inverse_transform(forecast_scaled)[0][0]
            forecast_mean = np.repeat(forecast_mean, periode_prediksi)
            
            # Confidence interval (simulasi)
            conf_int = np.array([forecast_mean*0.9, forecast_mean*1.1])
        except Exception as e:
            st.error(f"Gagal membuat model LSTM: {str(e)}")
            return

    # Format hasil prediksi
    future_dates = pd.date_range(start=df_history.index.max() + pd.DateOffset(months=1), 
                                 periods=periode_prediksi, 
                                 freq='MS')
    
    df_forecast = pd.DataFrame({
        'bulan': future_dates,
        'prediksi': forecast_mean,
        'lower_ci': conf_int[0],
        'upper_ci': conf_int[1]
    }).set_index('bulan')

    # Visualisasi
    df_combined = pd.DataFrame(index=df_history.index.union(df_forecast.index))
    df_combined['historis'] = df_history['total_keluar']
    df_combined['prediksi'] = np.nan
    df_combined.loc[df_forecast.index, 'prediksi'] = df_forecast['prediksi']
    df_combined['lower_ci'] = np.nan
    df_combined.loc[df_forecast.index, 'lower_ci'] = df_forecast['lower_ci']
    df_combined['upper_ci'] = np.nan
    df_combined.loc[df_forecast.index, 'upper_ci'] = df_forecast['upper_ci']

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_combined.index, y=df_combined['historis'],
                             mode='lines', name='Historis',
                             line=dict(color='#2ECC71', width=2)))
    fig.add_trace(go.Scatter(x=df_combined.index, y=df_combined['prediksi'],
                             mode='lines', name='Prediksi',
                             line=dict(color='#3498DB', width=2, dash='dash')))
    fig.add_trace(go.Scatter(
        x=df_combined.index.tolist() + df_combined.index.tolist()[::-1],
        y=df_combined['upper_ci'].tolist() + df_combined['lower_ci'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(52, 152, 219, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name=f'Interval {model_confidence}%'
    ))
    fig.update_layout(
        title=f'Prediksi Penggunaan {produk_pilihan} ({model_choice})',
        xaxis_title='Bulan',
        yaxis_title='Jumlah',
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tabel prediksi
    df_forecast_display = df_forecast.reset_index()
    df_forecast_display.columns = ['Bulan', 'Prediksi', 'Batas Bawah', 'Batas Atas']
    df_forecast_display['Bulan'] = df_forecast_display['Bulan'].dt.strftime('%B %Y')
    df_forecast_display = df_forecast_display.applymap(lambda x: max(0, round(x)) if isinstance(x, (int, float)) else x)

    st.subheader("📋 Tabel Prediksi")
    st.dataframe(
        df_forecast_display.style.format({
            'Prediksi': '{:,.0f}',
            'Batas Bawah': '{:,.0f}',
            'Batas Atas': '{:,.0f}'
        }),
        use_container_width=True
    )

    conn.close()