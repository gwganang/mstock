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
        col1, col2, col3 = st.columns([2, 1, 1])
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
                1, 12, 12,
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

    if len(df_history) < 3:
        st.warning(f"Data historis untuk '{produk_pilihan}' tidak cukup. Minimal 3 bulan data diperlukan.", icon="⚠️")
        st.info("Tambahkan transaksi keluar untuk mendapatkan data yang cukup.")
        return

    # Proses data historis
    df_history['bulan'] = pd.to_datetime(df_history['bulan'] + '-01')
    df_history.set_index('bulan', inplace=True)
    date_range = pd.date_range(start=df_history.index.min(), end=df_history.index.max(), freq='MS')
    df_history = df_history.reindex(date_range, fill_value=0)

    # Tampilkan data historis
    st.subheader("📊 Data Historis Penggunaan")
    fig_history = px.line(
        df_history,
        y='total_keluar',
        title=f'Penggunaan Historis {produk_pilihan}',
        labels={'total_keluar': 'Jumlah', 'bulan': 'Bulan'},
        markers=True,
        template='plotly_white'
    )
    fig_history.update_traces(line_color='#2ECC71', marker=dict(size=8))
    st.plotly_chart(fig_history, use_container_width=True)

    # Pemodelan ARIMA
    with st.spinner('Memproses prediksi...'):
        try:
            # Fitting model ARIMA
            model = ARIMA(df_history['total_keluar'], order=(1,1,1))
            model_fit = model.fit()

            # Generate prediksi
            last_date = df_history.index.max()
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                         periods=periode_prediksi, 
                                         freq='MS')
            
            forecast = model_fit.get_forecast(steps=periode_prediksi)
            forecast_mean = forecast.predicted_mean
            forecast_ci = forecast.conf_int(alpha=(100 - model_confidence)/100)

            # Format hasil prediksi
            df_forecast = pd.DataFrame({
                'bulan': future_dates,
                'prediksi': forecast_mean.values,
                'lower_ci': forecast_ci.iloc[:, 0].values,
                'upper_ci': forecast_ci.iloc[:, 1].values
            }).set_index('bulan')

            # === PERBAIKAN UTAMA: GABUNG DATA DENGAN UNION ===
            combined_index = df_history.index.union(df_forecast.index)
            df_combined = pd.DataFrame(index=combined_index)
            df_combined['historis'] = df_history['total_keluar']
            df_combined['prediksi'] = np.nan
            df_combined.loc[df_forecast.index, 'prediksi'] = df_forecast['prediksi']
            df_combined['lower_ci'] = np.nan
            df_combined.loc[df_forecast.index, 'lower_ci'] = df_forecast['lower_ci']
            df_combined['upper_ci'] = np.nan
            df_combined.loc[df_forecast.index, 'upper_ci'] = df_forecast['upper_ci']

            # Visualisasi
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_combined.index, y=df_combined['historis'],
                                     mode='lines+markers', name='Historis',
                                     line=dict(color='#2ECC71')))
            fig.add_trace(go.Scatter(x=df_combined.index, y=df_combined['prediksi'],
                                     mode='lines+markers', name='Prediksi',
                                     line=dict(color='#3498DB', dash='dash')))
            
            # Confidence interval
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

        except Exception as e:
            st.error(f"Terjadi kesalahan: {str(e)}")
            st.info("Pastikan data memadai dan parameter valid.")

    conn.close()