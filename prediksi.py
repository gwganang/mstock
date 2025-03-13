import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Judul halaman
    st.header("📈 Prediksi Stok", divider="green")

    # Koneksi ke database
    conn = sqlite3.connect('data/stok.db')
    c = conn.cursor()

    # Pilih produk
    produk = c.execute("SELECT id, nama FROM produk").fetchall()
    if not produk:
        st.warning("Tidak ada produk tersedia. Silakan tambah produk terlebih dahulu.", icon="⚠️")
        conn.close()
        return
    daftar_produk = [p[1] for p in produk]
    produk_pilihan = st.selectbox("Pilih Produk untuk Prediksi", daftar_produk)
    produk_id = [p[0] for p in produk if p[1] == produk_pilihan][0]

    # 1. Data Transaksi Barang Terpilih
    st.subheader("Data Transaksi Keluar Barang Terpilih")
    df_transaksi = pd.read_sql_query(
        f"SELECT tanggal, jumlah FROM transaksi_keluar WHERE produk_id = {produk_id} ORDER BY tanggal",
        conn,
        parse_dates=['tanggal']
    )
    conn.close()

    if df_transaksi.empty:
        st.warning("Tidak ada data transaksi keluar untuk produk ini.", icon="⚠️")
        return

    # Agregasi data per bulan (3 tahun terakhir)
    df_transaksi.set_index('tanggal', inplace=True)
    df_monthly = df_transaksi.resample('M').sum().reset_index()
    st.dataframe(df_monthly)

    # 2. Grafik Data Transaksi Barang Terpilih
    st.subheader("Grafik Data Transaksi Keluar")
    fig_transaksi = px.line(df_monthly, x='tanggal', y='jumlah', title='Transaksi Keluar Bulanan')
    st.plotly_chart(fig_transaksi, use_container_width=True)

    # 3. Uji Stasioneritas dan Differensiasi
    st.subheader("Uji Stasioneritas (ADF Test)")
    result = adfuller(df_monthly['jumlah'])
    st.write(f"ADF Statistic: {result[0]}")
    st.write(f"p-value: {result[1]}")
    if result[1] > 0.05:
        st.write("Data tidak stasioner, dilakukan differensiasi.")
        df_diff = df_monthly['jumlah'].diff().dropna()
        # 4. Grafik Data Differensiasi
        st.subheader("Grafik Data Setelah Differensiasi")
        fig_diff = px.line(df_diff, title='Data Setelah Differensiasi')
        st.plotly_chart(fig_diff, use_container_width=True)
    else:
        st.write("Data sudah stasioner, asumsi metode ARIMA terpenuhi.")
        df_diff = df_monthly['jumlah']

    # 5. Plot ACF dan PACF
    st.subheader("Plot Autokorelasi (ACF) dan Autokorelasi Parsial (PACF)")
    fig_acf, ax_acf = plt.subplots()
    plot_acf(df_diff, lags=20, ax=ax_acf)
    st.pyplot(fig_acf)

    fig_pacf, ax_pacf = plt.subplots()
    plot_pacf(df_diff, lags=20, ax=ax_pacf)
    st.pyplot(fig_pacf)

    # 6. Estimasi Model ARIMA
    st.subheader("Estimasi Model ARIMA")
    models = [(1,1,1), (0,1,1), (1,1,2)]  # Contoh kombinasi model
    model_results = {}
    for order in models:
        try:
            model = ARIMA(df_monthly['jumlah'], order=order)
            results = model.fit()
            model_results[order] = results
            st.write(f"Model {order} berhasil diestimasi.")
        except Exception as e:
            st.write(f"Model {order} gagal: {str(e)}")

    # 7. Pemilihan Model Terbaik
    st.subheader("Pemilihan Model Terbaik")
    best_model = None
    best_mse = np.inf
    for order, results in model_results.items():
        forecast = results.forecast(steps=12)
        # Gunakan data terakhir untuk evaluasi sederhana (pseudo-MSE)
        if len(df_monthly) >= 12:
            actual = df_monthly['jumlah'][-12:]
            mse = np.mean((forecast - actual) ** 2)
        else:
            mse = results.mse  # Gunakan MSE dari model jika data kurang
        st.write(f"Model {order}: MSE = {mse}")
        if mse < best_mse:
            best_mse = mse
            best_model = results

    if best_model:
        st.write(f"Model Terbaik: Order {best_model.model.order}, MSE: {best_mse}")
    else:
        st.error("Tidak ada model yang cocok ditemukan.", icon="❌")
        return

    # 8. Peramalan
    st.subheader("Peramalan untuk 12 Bulan ke Depan")
    forecast = best_model.forecast(steps=12)
    forecast_dates = pd.date_range(start=df_monthly['tanggal'].iloc[-1], periods=13, freq='M')[1:]
    df_forecast = pd.DataFrame({'tanggal': forecast_dates, 'prediksi': forecast})
    st.dataframe(df_forecast)

    # Grafik Peramalan
    fig_forecast = px.line(df_forecast, x='tanggal', y='prediksi', title='Prediksi Pengadaan Barang')
    st.plotly_chart(fig_forecast, use_container_width=True)

if __name__ == "__main__":
    main()