import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
import matplotlib.pyplot as plt
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

def main():
    st.header("📈 Prediksi Stok dengan ARIMA", divider="green")
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
        col1, col2, col3 = st.columns([3, 1, 1])
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

    if len(df_history) < 12:
        st.error("Data tidak cukup! Minimal 12 bulan data diperlukan untuk prediksi ARIMA")
        return

    # Proses data historis
    df_history['bulan'] = pd.to_datetime(df_history['bulan'] + '-01')
    df_history.set_index('bulan', inplace=True)
    date_range = pd.date_range(start=df_history.index.min(), end=df_history.index.max(), freq='MS')
    df_history = df_history.reindex(date_range, method='ffill').fillna(0)

    # Analisis Stasioneritas
    with st.expander("📊 Analisis Stasioneritas"):
        result = adfuller(df_history['total_keluar'])
        st.write(f'p-value: {result[1]:.4f}')
        if result[1] > 0.05:
            st.warning("Data tidak stasioner! Akan dilakukan differencing otomatis", icon="⚠️")
        else:
            st.success("Data stasioner", icon="✅")

        # Plot ACF dan PACF
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        plot_acf(df_history['total_keluar'], ax=ax1, lags=12)
        plot_pacf(df_history['total_keluar'], ax=ax2, lags=12)
        st.pyplot(fig)

    # Pemodelan ARIMA
    with st.spinner('Melakukan pemodelan ARIMA...'):
        try:
            # Auto ARIMA untuk pemilihan parameter optimal
            auto_model = auto_arima(
                df_history['total_keluar'],
                seasonal=False,
                stepwise=True,
                trace=True,
                error_action='ignore',
                suppress_warnings=True
            )
            
            # Model final dengan parameter terbaik
            model = ARIMA(
                df_history['total_keluar'],
                order=auto_model.order
            )
            results = model.fit()

            # Prediksi
            forecast = results.get_forecast(steps=periode_prediksi)
            forecast_mean = forecast.predicted_mean
            conf_int = forecast.conf_int(alpha=(100 - model_confidence)/100)

            # Format hasil
            future_dates = pd.date_range(
                start=df_history.index.max() + pd.DateOffset(months=1),
                periods=periode_prediksi,
                freq='MS'
            )
            
            df_forecast = pd.DataFrame({
                'bulan': future_dates,
                'prediksi': forecast_mean,
                'lower_ci': conf_int.iloc[:, 0],
                'upper_ci': conf_int.iloc[:, 1]
            }).set_index('bulan')

        except Exception as e:
            st.error(f"Gagal membuat model ARIMA: {str(e)}")
            return

    # Evaluasi Model
    train_size = int(len(df_history) * 0.8)
    train, test = df_history[:train_size], df_history[train_size:]
    forecast_test = results.get_forecast(steps=len(test)).predicted_mean
    mae = np.mean(np.abs(forecast_test - test['total_keluar']))
    rmse = np.sqrt(np.mean((forecast_test - test['total_keluar'])**2))

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
        title=f'Prediksi Penggunaan {produk_pilihan}',
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

    # Metrik evaluasi
    st.subheader("🔍 Evaluasi Model")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("MAE", f"{mae:.2f}", help="Mean Absolute Error")
    with col2:
        st.metric("RMSE", f"{rmse:.2f}", help="Root Mean Squared Error")

    # Ringkasan model
    with st.expander("📚 Ringkasan Model ARIMA"):
        st.write(f"Parameter ARIMA terpilih: {auto_model.order}")
        st.code(results.summary().as_text())

    conn.close()