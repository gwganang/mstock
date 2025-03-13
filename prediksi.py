import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

plt.style.use('ggplot')

def main():
    st.header("📈 Prediksi Stok dengan ARIMA", divider="green")
    conn = sqlite3.connect('data/stok.db')
    c = conn.cursor()

    # Ambil daftar produk
    produk = c.execute("SELECT id, nama FROM produk").fetchall()
    if not produk:
        st.warning("Tidak ada produk tersedia!", icon="⚠️")
        return

    # Pilih produk
    selected_produk_id = st.selectbox(
        "📦 Pilih Produk",
        options=[p[0] for p in produk],
        format_func=lambda x: next(p[1] for p in produk if p[0] == x),
        help="Pilih produk untuk prediksi"
    )
    selected_produk_nama = next(p[1] for p in produk if p[0] == selected_produk_id)

    # Ambil data transaksi keluar 3 tahun terakhir
    df = pd.read_sql_query('''
        SELECT 
            tanggal AS Tanggal,
            jumlah AS Jumlah
        FROM transaksi_keluar
        WHERE produk_id = ? AND tanggal >= date('now', '-3 years')
        ORDER BY tanggal ASC
    ''', conn, params=(selected_produk_id,))

    if df.empty:
        st.warning(f"Tidak ada data transaksi untuk {selected_produk_nama} dalam 3 tahun terakhir!", icon="⚠️")
        return

    # Konversi ke time series bulanan
    try:
        df['Tanggal'] = pd.to_datetime(df['Tanggal'])
        df.set_index('Tanggal', inplace=True)
        monthly_data = df.resample('ME').sum()  # Menggunakan ME (Month End)
    except Exception as e:
        st.error(f"Error konversi data: {str(e)}", icon="❌")
        return

    # Validasi panjang data
    if len(monthly_data) < 12:
        st.warning("Data transaksi tidak cukup untuk analisis yang reliable. Minimal 12 bulan data diperlukan", 
                   icon="⚠️")
        return

    # Tampilkan data
    st.subheader(f"Data Transaksi {selected_produk_nama} (3 Tahun Terakhir)")
    st.dataframe(monthly_data, use_container_width=True)

    # Plot data asli
    st.subheader("Grafik Transaksi")
    fig = px.line(
        monthly_data,
        y='Jumlah',
        title='Pola Transaksi Bulanan',
        labels={'Jumlah': 'Jumlah Transaksi', 'Tanggal': 'Bulan'},
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Uji stasioneritas
    st.subheader("Uji Stasioneritas")
    result = adfuller(monthly_data['Jumlah'].dropna())
    st.write(f'ADF Statistic: {result[0]:.4f}')
    st.write(f'p-value: {result[1]:.4f}')
    st.write('Critical Values:')
    for key, value in result[4].items():
        st.write(f'  {key}: {value:.4f}')

    # Differencing jika tidak stasioner
    if result[1] > 0.05:
        st.warning("Data tidak stasioner! Melakukan differencing...", icon="⚠️")
        differenced = monthly_data.diff().dropna()
        
        # Validasi data differencing
        if len(differenced) < 12:
            st.warning("Data terlalu sedikit setelah differencing", icon="⚠️")
            return
        if differenced['Jumlah'].var() == 0:
            st.warning("Data tidak memiliki variasi setelah differencing", icon="⚠️")
            return
            
        # Plot differenced data
        st.subheader("Grafik Data Differensiasi")
        fig_diff = px.line(
            differenced,
            y='Jumlah',
            title='Data Setelah Differencing',
            labels={'Jumlah': 'Perubahan Jumlah', 'Tanggal': 'Bulan'},
            template='plotly_white'
        )
        st.plotly_chart(fig_diff, use_container_width=True)
        
        # Uji stasioneritas setelah differencing
        result_diff = adfuller(differenced['Jumlah'].dropna())
        st.write("Hasil Uji Differencing:")
        st.write(f'ADF Statistic: {result_diff[0]:.4f}')
        st.write(f'p-value: {result_diff[1]:.4f}')
    else:
        differenced = monthly_data

    # Fungsi helper untuk plot dengan penanganan error
    def safe_plot(plot_func, data_series, *args, **kwargs):
        try:
            fig, ax = plt.subplots(figsize=(10,4))
            sample_size = len(data_series)
            max_allowed_lags = (sample_size - 1) // 2 - 1
            max_lag = min(20, max_allowed_lags)
            
            if max_lag < 1:
                st.warning("Data terlalu pendek untuk analisis PACF", icon="⚠️")
                return None
                
            plot_func(data_series, lags=max_lag, ax=ax, **kwargs)
            return fig
        except Exception as e:
            st.error(f"Plot gagal: {str(e)}", icon="❌")
            return None

    # Plot ACF dan PACF
    st.subheader("Analisis ACF dan PACF")
    col1, col2 = st.columns(2)
    with col1:
        fig_acf = safe_plot(plot_acf, differenced['Jumlah'])
        if fig_acf:
            st.pyplot(fig_acf)
    with col2:
        fig_pacf = safe_plot(plot_pacf, differenced['Jumlah'])
        if fig_pacf:
            st.pyplot(fig_pacf)

    # Pencarian parameter ARIMA terbaik
    st.subheader("Pemilihan Model ARIMA")
    p_values = range(0, 3)
    q_values = range(0, 3)
    d = 1 if result[1] > 0.05 else 0

    best_mse = np.inf
    best_order = None
    results = []

    with st.spinner("Mencari model terbaik..."):
        for p in p_values:
            for q in q_values:
                try:
                    model = ARIMA(
                        monthly_data,
                        order=(p, d, q),
                        enforce_invertibility=True,
                        enforce_stationarity=True
                    )
                    results_fit = model.fit(
                        method='innovations_mle',
                        low_memory=True,
                        maxiter=1000  # Hapus parameter 'disp'
                    )
                    mse = mean_squared_error(
                        monthly_data['Jumlah'][d:], 
                        results_fit.fittedvalues
                    )
                    results.append({
                        'order': f'({p},{d},{q})',
                        'MSE': mse,
                        'AIC': results_fit.aic
                    })
                    if mse < best_mse:
                        best_mse = mse
                        best_order = (p, d, q)
                except Exception as e:
                    st.warning(f"Model ({p},{d},{q}) gagal: {str(e)}")
                    continue

    # Tampilkan hasil perbandingan
    if results:
        st.dataframe(
            pd.DataFrame(results).sort_values('MSE'),
            column_config={
                "order": "Model Order",
                "MSE": st.column_config.NumberColumn("MSE", format="%.2f"),
                "AIC": st.column_config.NumberColumn("AIC", format="%.2f")
            },
            use_container_width=True
        )

        # Estimasi model terbaik
        st.subheader(f"Model Terpilih: ARIMA{best_order}")
        try:
            final_model = ARIMA(
                monthly_data,
                order=best_order,
                enforce_invertibility=True,
                enforce_stationarity=True
            )
            final_fit = final_model.fit(
                method='innovations_mle',
                low_memory=True,
                maxiter=1000  # Hapus parameter 'disp'
            )
            st.text(final_fit.summary())

            # Forecast 12 bulan ke depan
            st.subheader("Peramalan untuk 12 Bulan ke Depan")
            forecast = final_fit.get_forecast(steps=12)
            forecast_index = pd.date_range(
                start=monthly_data.index[-1] + timedelta(days=1),
                periods=12,
                freq='ME'  # Menggunakan ME (Month End)
            )

            # Plot forecast
            fig_forecast = px.line(
                title='Peramalan Stok',
                template='plotly_white'
            )
            fig_forecast.add_scatter(
                x=monthly_data.index,
                y=monthly_data['Jumlah'],
                name='Data Historis'
            )
            fig_forecast.add_scatter(
                x=forecast_index,
                y=forecast.predicted_mean,
                name='Peramalan',
                line=dict(dash='dash')
            )
            fig_forecast.add_scatter(
                x=forecast_index,
                y=forecast.conf_int().iloc[:, 0],
                line=dict(color='rgba(255,0,0,0.2)'),
                name='Interval Kepercayaan'
            )
            fig_forecast.add_scatter(
                x=forecast_index,
                y=forecast.conf_int().iloc[:, 1],
                line=dict(color='rgba(255,0,0,0.2)'),
                name='Interval Kepercayaan',
                showlegend=False
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
            
        except Exception as e:
            st.error(f"Gagal membuat peramalan: {str(e)}", icon="❌")
    else:
        st.warning("Tidak ada model ARIMA yang valid ditemukan", icon="⚠️")

    conn.close()