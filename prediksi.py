import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")
plt.style.use('ggplot')

# Konfigurasi Streamlit
st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.header("🔮 Prediksi Stok Cerdas (ARIMA/SARIMA)", divider="rainbow")
    conn = sqlite3.connect('data/stok.db')
    c = conn.cursor()

    # === 1. Pemilihan Produk ===
    produk = c.execute("SELECT id, nama FROM produk").fetchall()
    if not produk:
        st.warning("⚠️ Tidak ada produk tersedia! Silakan tambahkan produk di menu Produk", icon="⚠️")
        return

    selected_produk_id = st.selectbox(
        "📦 Pilih Produk untuk Prediksi",
        options=[p[0] for p in produk],
        format_func=lambda x: next(p[1] for p in produk if p[0] == x),
        help="Pilih produk yang ingin diprediksi"
    )
    selected_produk_nama = next(p[1] for p in produk if p[0] == selected_produk_id)

    # === 2. Ekstraksi Data ===
    df = pd.read_sql_query('''
        SELECT tanggal AS Tanggal, jumlah AS Jumlah
        FROM transaksi_keluar
        WHERE produk_id = ? AND tanggal >= date('now', '-3 years')
        ORDER BY tanggal ASC
    ''', conn, params=(selected_produk_id,))

    if df.empty:
        st.warning(f"❌ Tidak ada data transaksi untuk {selected_produk_nama} dalam 3 tahun terakhir!", icon="⚠️")
        return

    # === 3. Pra-pemrosesan Data ===
    try:
        df['Tanggal'] = pd.to_datetime(df['Tanggal'])
        df.set_index('Tanggal', inplace=True)
        monthly_data = df.resample('ME').sum()  # Month End
    except Exception as e:
        st.error(f"❌ Error konversi data: {str(e)}", icon="❌")
        return

    if len(monthly_data) < 12:
        st.warning("⚠️ Data kurang dari 12 bulan! Prediksi mungkin tidak akurat", icon="⚠️")
    else:
        st.success(f"✅ Data memadai ({len(monthly_data)} bulan)", icon="✅")

    # === 4. Analisis Stasioneritas ===
    with st.expander("📊 Analisis Stasioneritas", expanded=False):
        result = adfuller(monthly_data['Jumlah'].dropna())
        st.write("#### Uji Dickey-Fuller:")
        st.write(f"- ADF Statistic: {result[0]:.4f}")
        st.write(f"- p-value: {result[1]:.4f}")
        
        if result[1] > 0.05:
            st.warning("⚠️ Data tidak stasioner! Akan dilakukan differencing", icon="⚠️")
            differenced = monthly_data.diff().dropna()
            st.write("#### Setelah Differencing:")
            result_diff = adfuller(differenced['Jumlah'].dropna())
            st.write(f"- ADF Statistic: {result_diff[0]:.4f}")
            st.write(f"- p-value: {result_diff[1]:.4f}")
        else:
            st.success("✅ Data sudah stasioner", icon="✅")
            differenced = monthly_data

    # === 5. Visualisasi Dinamis ===
    st.subheader("📈 Visualisasi Data")
    tab1, tab2, tab3 = st.tabs(["Data Historis", "ACF/PACF", "Differencing"])

    with tab1:
        fig = px.line(
            monthly_data,
            y='Jumlah',
            title='Pola Transaksi Bulanan',
            labels={'Jumlah': 'Jumlah Transaksi', 'index': 'Bulan'},
            template='plotly_dark'
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        def safe_plot(data, plot_type):
            try:
                max_lag = min(20, len(data)//2 - 1)
                if max_lag < 1:
                    st.warning("⚠️ Data terlalu pendek untuk analisis", icon="⚠️")
                    return
                
                fig, ax = plt.subplots(figsize=(10,4))
                if plot_type == 'acf':
                    plot_acf(data, lags=max_lag, ax=ax)
                else:
                    plot_pacf(data, lags=max_lag, ax=ax)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"❌ Error plotting: {str(e)}", icon="❌")

        col1, col2 = st.columns(2)
        with col1:
            st.write("#### ACF Plot")
            safe_plot(differenced['Jumlah'], 'acf')
        with col2:
            st.write("#### PACF Plot")
            safe_plot(differenced['Jumlah'], 'pacf')

    with tab3:
        if 'differenced' in locals():
            fig = px.line(
                differenced,
                y='Jumlah',
                title='Data Setelah Differencing',
                labels={'Jumlah': 'Perubahan Jumlah', 'index': 'Bulan'},
                template='plotly_dark'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Tidak ada proses differencing dilakukan", icon="ℹ️")

    # === 6. Pemodelan Otomatis ===
    st.subheader("🤖 Pemilihan Model Otomatis")
    with st.spinner("⏳ Melakukan Auto-ARIMA..."):
        try:
            model = auto_arima(
                monthly_data,
                seasonal=True,
                m=12,  # Bulanan
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )
            
            best_order = model.order
            best_seasonal = model.seasonal_order
            st.success(f"✅ Model Terpilih: ARIMA{best_order} x SARIMA{best_seasonal}", icon="✅")
            
        except Exception as e:
            st.error(f"❌ Gagal melakukan Auto-ARIMA: {str(e)}", icon="❌")
            return

    # === 7. Pelatihan Model ===
    try:
        model.fit(monthly_data)
        forecast = model.predict(n_periods=12)
        forecast_index = pd.date_range(
            start=monthly_data.index[-1] + timedelta(days=1),
            periods=12,
            freq='ME'
        )
        
        # Evaluasi
        mae = mean_absolute_error(
            monthly_data['Jumlah'][-12:],
            model.predict_in_sample()[-12:]
        )
        
    except Exception as e:
        st.error(f"❌ Error pelatihan model: {str(e)}", icon="❌")
        return

    # === 8. Visualisasi Prediksi ===
    st.subheader("🔮 Hasil Prediksi 12 Bulan")
    forecast_df = pd.DataFrame({
        'Prediksi': forecast,
        'Interval Bawah': model.predict(n_periods=12, return_conf_int=True)[1][:,0],
        'Interval Atas': model.predict(n_periods=12, return_conf_int=True)[1][:,1]
    }, index=forecast_index)
    
    fig = px.line(
        title='Peramalan Stok',
        template='plotly_dark'
    )
    fig.add_scatter(
        x=monthly_data.index,
        y=monthly_data['Jumlah'],
        name='Data Historis'
    )
    fig.add_scatter(
        x=forecast_df.index,
        y=forecast_df['Prediksi'],
        name='Prediksi',
        line=dict(dash='dash', color='yellow')
    )
    fig.add_scatter(
        x=forecast_df.index,
        y=forecast_df['Interval Atas'],
        line=dict(color='rgba(0,255,0,0.2)'),
        name='Interval Kepercayaan'
    )
    fig.add_scatter(
        x=forecast_df.index,
        y=forecast_df['Interval Bawah'],
        line=dict(color='rgba(0,255,0,0.2)'),
        name='Interval Kepercayaan',
        fill='tonexty'
    )
    st.plotly_chart(fig, use_container_width=True)

    # === 9. Ringkasan Model ===
    st.subheader("📋 Ringkasan Model")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("MAE (Error Rata-rata)", f"{mae:.2f}")
    with col2:
        st.metric("AIC", f"{model.aic():.2f}")

    st.write("#### Parameter Model:")
    st.write(f"- ARIMA Order: {best_order}")
    st.write(f"- Seasonal Order: {best_seasonal}")

    # === 10. Rekomendasi Pengadaan ===
    st.subheader("🛒 Rekomendasi Pengadaan")
    average_demand = forecast.mean()
    st.info(f"""
        Berdasarkan prediksi:
        - Sediakan rata-rata **{average_demand:.0f} unit/bulan**
        - Siapkan buffer stok sebesar **{forecast.std():.0f} unit** 
          untuk mengantisipasi fluktuasi
    """, icon="💡")

    conn.close()

if __name__ == "__main__":
    main()