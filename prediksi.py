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
    
    # Container untuk kontrol prediksi
    with st.container():
        st.subheader("🔮 Parameter Prediksi")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Pilih Produk
            daftar_produk = [p[1] for p in produk]
            produk_pilihan = st.selectbox(
                "Produk yang akan diprediksi",
                daftar_produk,
                format_func=lambda x: f"📦 {x}",
                help="Pilih produk yang ingin diprediksi stoknya"
            )
            produk_id = next((p[0] for p in produk if p[1] == produk_pilihan), None)
        
        with col2:
            # Periode prediksi (dalam bulan)
            periode_prediksi = st.slider(
                "Periode Prediksi (bulan)",
                min_value=1,
                max_value=12,
                value=12,
                help="Jumlah bulan yang akan diprediksi ke depan"
            )
        
        with col3:
            # Parameter ARIMA
            model_confidence = st.slider(
                "Tingkat Kepercayaan (%)",
                min_value=80,
                max_value=99,
                value=95,
                help="Tingkat kepercayaan untuk interval prediksi"
            )
    
    # Ambil data historikal
    # Konversi transaksi keluar menjadi time series bulanan
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
        st.warning(f"Data historis untuk produk '{produk_pilihan}' tidak cukup untuk analisis. Minimal diperlukan data 3 bulan.", icon="⚠️")
        # Tampilkan saran untuk user
        st.info("Untuk menggunakan fitur prediksi, pastikan Anda memiliki data transaksi keluar yang cukup. Tambahkan beberapa transaksi keluar terlebih dahulu.", icon="ℹ️")
        return
    
    # Parse tanggal
    df_history['bulan'] = pd.to_datetime(df_history['bulan'] + '-01')
    df_history.set_index('bulan', inplace=True)
    
    # Deteksi missing periods dan isi dengan 0
    date_range = pd.date_range(start=df_history.index.min(), end=df_history.index.max(), freq='MS')
    df_history = df_history.reindex(date_range, fill_value=0)
    
    # Menampilkan data historis
    st.subheader("📜 Data Historis Penggunaan")
    
    # Metrics overview
    with st.container():
        col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
        with col_metrics1:
            total_history = df_history['total_keluar'].sum()
            st.metric(
                "Total Penggunaan Historis",
                f"{int(total_history):,}",
                help="Total penggunaan dari seluruh data historis"
            )
        with col_metrics2:
            avg_monthly = df_history['total_keluar'].mean()
            st.metric(
                "Rata-rata Bulanan",
                f"{int(avg_monthly):,}",
                help="Rata-rata penggunaan per bulan"
            )
        with col_metrics3:
            std_monthly = df_history['total_keluar'].std()
            st.metric(
                "Std. Deviasi Bulanan",
                f"{int(std_monthly):,}" if not np.isnan(std_monthly) else "N/A",
                help="Standar deviasi penggunaan bulanan"
            )
    
    # Visualisasi data historis
    fig_history = px.line(
        df_history, 
        y='total_keluar',
        title=f'Penggunaan Historis {produk_pilihan}',
        labels={'total_keluar': 'Jumlah Penggunaan', 'bulan': 'Bulan'},
        markers=True,
        template='plotly_white'
    )
    
    fig_history.update_traces(line_color='#2ECC71', marker=dict(size=8, color='#28B463'))
    fig_history.update_layout(xaxis_title="Bulan", yaxis_title="Jumlah")
    st.plotly_chart(fig_history, use_container_width=True)
    
    # Tabel Data Historis
    st.write("Tabel Data Historis Bulanan")
    df_history_display = df_history.reset_index()
    df_history_display.columns = ['Bulan', 'Jumlah Penggunaan']
    df_history_display['Bulan'] = df_history_display['Bulan'].dt.strftime('%B %Y')
    
    # Styling untuk tabel
    styled_history = df_history_display.style.format({
        'Jumlah Penggunaan': '{:,.0f}'
    })
    
    st.dataframe(
        styled_history,
        use_container_width=True,
        hide_index=True
    )
    
    # Pemodelan ARIMA dan Prediksi
    st.subheader("🔮 Analisis dan Prediksi ARIMA")
    
    with st.expander("ℹ️ Informasi tentang ARIMA", expanded=False):
        st.write("""
        **ARIMA (Autoregressive Integrated Moving Average)** adalah model statistik yang menganalisis dan memproyeksikan data time series. Model ini terdiri dari tiga komponen:
        
        1. **AR (Autoregressive)**: Penggunaan nilai observasi sebelumnya sebagai input untuk prediksi
        2. **I (Integrated)**: Berapa kali data perlu didifferensiasi untuk mencapai stasioneritas
        3. **MA (Moving Average)**: Penggunaan error prediksi sebelumnya dalam model
        
        Untuk mendapatkan hasil terbaik, model ARIMA memerlukan data historis yang cukup (minimal 2 tahun untuk prediksi tahunan).
        """)
    
    # Analisis stasioneritas (untuk pengguna teknis)
    with st.expander("🔍 Analisis Stasioneritas", expanded=False):
        st.write("Sebelum menggunakan ARIMA, data perlu diuji untuk stasioneritas:")
        
        # Dickey-Fuller Test
        result = adfuller(df_history['total_keluar'].dropna())
        st.write('Hasil Uji Augmented Dickey-Fuller:')
        st.write(f'Statistik ADF: {result[0]:.4f}')
        st.write(f'p-value: {result[1]:.4f}')
        
        for key, value in result[4].items():
            st.write(f'Nilai Kritis {key}: {value:.4f}')
        
        if result[1] <= 0.05:
            st.success("Data stasioner (p-value ≤ 0.05)")
        else:
            st.warning("Data tidak stasioner (p-value > 0.05)")
        
        # Visualisasi ACF/PACF
        st.write("Grafik ACF dan PACF untuk penentuan parameter ARIMA:")
        
        # Buat subplot untuk ACF dan PACF
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot ACF dan PACF
        plot_acf(df_history['total_keluar'].dropna(), ax=ax1, lags=12)
        plot_pacf(df_history['total_keluar'].dropna(), ax=ax2, lags=12)
        
        plt.tight_layout()
        
        # Konversi plot Matplotlib ke Streamlit
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        
        st.image(buf)
    
    # Auto ARIMA fitting
    with st.spinner('Melakukan pemodelan ARIMA dan prediksi...'):
        # Fitting model ARIMA
        # Untuk kesederhanaan, kita gunakan (1,1,1) sebagai default
        # Untuk aplikasi produksi, sebaiknya gunakan auto_arima dari pmdarima
        try:
            model = ARIMA(df_history['total_keluar'], order=(1,1,1))
            model_fit = model.fit()
            
            # Prediksi untuk periode mendatang
            last_date = df_history.index.max()
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                          periods=periode_prediksi, 
                                          freq='MS')
            
            # Prediksi
            forecast = model_fit.get_forecast(steps=periode_prediksi)
            forecast_mean = forecast.predicted_mean
            forecast_ci = forecast.conf_int(alpha=(100-model_confidence)/100)
            
            # DataFrame untuk prediksi
            df_forecast = pd.DataFrame({
                'bulan': future_dates,
                'prediksi': forecast_mean.values,
                'lower_ci': forecast_ci.iloc[:, 0].values,
                'upper_ci': forecast_ci.iloc[:, 1].values
            })
            df_forecast.set_index('bulan', inplace=True)
            
            # Menggabungkan data historis dan prediksi untuk visualisasi
            df_combined = pd.DataFrame(index=pd.concat([df_history.index, df_forecast.index]))
            df_combined['historis'] = df_history['total_keluar']
            df_combined['prediksi'] = np.nan
            df_combined.loc[df_forecast.index, 'prediksi'] = df_forecast['prediksi']
            df_combined['lower_ci'] = np.nan
            df_combined['upper_ci'] = np.nan
            df_combined.loc[df_forecast.index, 'lower_ci'] = df_forecast['lower_ci']
            df_combined.loc[df_forecast.index, 'upper_ci'] = df_forecast['upper_ci']
            
            # Visualisasi prediksi
            st.subheader(f"🔮 Prediksi Penggunaan {periode_prediksi} Bulan Mendatang")
            
            # Plot dengan plotly
            fig = go.Figure()
            
            # Data historis
            fig.add_trace(go.Scatter(
                x=df_combined.index,
                y=df_combined['historis'],
                mode='lines+markers',
                name='Data Historis',
                line=dict(color='#2ECC71', width=2),
                marker=dict(size=6)
            ))
            
            # Prediksi
            fig.add_trace(go.Scatter(
                x=df_combined.index,
                y=df_combined['prediksi'],
                mode='lines+markers',
                name='Prediksi',
                line=dict(color='#3498DB', width=2, dash='dash'),
                marker=dict(size=6)
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=df_combined.index.tolist() + df_combined.index.tolist()[::-1],
                y=df_combined['upper_ci'].tolist() + df_combined['lower_ci'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(52, 152, 219, 0.2)',
                line=dict(color='rgba(52, 152, 219, 0)'),
                name=f'Interval Kepercayaan {model_confidence}%'
            ))
            
            fig.update_layout(
                title=f'Prediksi Penggunaan {produk_pilihan}',
                xaxis_title='Bulan',
                yaxis_title='Jumlah Penggunaan',
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabel prediksi
            st.subheader("📋 Tabel Prediksi Penggunaan Bulanan")
            
            df_forecast_display = df_forecast.reset_index()
            df_forecast_display.columns = ['Bulan', 'Prediksi', 'Batas Bawah', 'Batas Atas']
            df_forecast_display['Bulan'] = df_forecast_display['Bulan'].dt.strftime('%B %Y')
            
            # Memastikan nilai prediksi tidak negatif
            df_forecast_display['Prediksi'] = df_forecast_display['Prediksi'].apply(lambda x: max(0, round(x)))
            df_forecast_display['Batas Bawah'] = df_forecast_display['Batas Bawah'].apply(lambda x: max(0, round(x)))
            df_forecast_display['Batas Atas'] = df_forecast_display['Batas Atas'].apply(lambda x: max(0, round(x)))
            
            # Tambahkan kolom untuk keperluan pengadaan
            df_forecast_display['Rekomendasi Pengadaan'] = df_forecast_display['Prediksi'].apply(lambda x: round(x * 1.1))  # 10% safety stock
            
            # Styling untuk tabel
            styled_forecast = df_forecast_display.style.format({
                'Prediksi': '{:,.0f}',
                'Batas Bawah': '{:,.0f}',
                'Batas Atas': '{:,.0f}',
                'Rekomendasi Pengadaan': '{:,.0f}'
            })
            
            st.dataframe(
                styled_forecast,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Bulan": "Bulan",
                    "Prediksi": st.column_config.NumberColumn(
                        "Prediksi Penggunaan",
                        format="%d",
                        help="Prediksi penggunaan per bulan"
                    ),
                    "Batas Bawah": st.column_config.NumberColumn(
                        f"Batas Bawah ({model_confidence}%)",
                        format="%d",
                        help=f"Batas bawah interval kepercayaan {model_confidence}%"
                    ),
                    "Batas Atas": st.column_config.NumberColumn(
                        f"Batas Atas ({model_confidence}%)",
                        format="%d",
                        help=f"Batas atas interval kepercayaan {model_confidence}%"
                    ),
                    "Rekomendasi Pengadaan": st.column_config.NumberColumn(
                        "Rekomendasi Pengadaan",
                        format="%d",
                        help="Rekomendasi jumlah pengadaan (+10% safety stock)"
                    )
                }
            )
            
            # Ringkasan prediksi
            st.subheader("📝 Ringkasan Prediksi")
            
            total_prediksi = df_forecast_display['Prediksi'].sum()
            
            col_sum1, col_sum2, col_sum3 = st.columns(3)
            with col_sum1:
                st.metric(
                    f"Total Prediksi ({periode_prediksi} bulan)",
                    f"{int(total_prediksi):,}",
                    help=f"Total prediksi penggunaan untuk {periode_prediksi} bulan ke depan"
                )
            with col_sum2:
                avg_forecast = df_forecast_display['Prediksi'].mean()
                st.metric(
                    "Rata-rata Bulanan",
                    f"{int(avg_forecast):,}",
                    delta=f"{int(avg_forecast - avg_monthly):+,}",
                    help="Perbandingan dengan rata-rata historis"
                )
            with col_sum3:
                total_procurement = df_forecast_display['Rekomendasi Pengadaan'].sum()
                st.metric(
                    "Rekomendasi Total Pengadaan",
                    f"{int(total_procurement):,}",
                    help="Rekomendasi total pengadaan dengan safety stock"
                )
            
            # Unduh laporan prediksi
            csv = df_forecast_display.to_csv(index=False)
            st.download_button(
                label="📥 Unduh Laporan Prediksi (CSV)",
                data=csv,
                file_name=f"prediksi_{produk_pilihan}_{datetime.now().strftime('%Y-%m-%d')}.csv",
                mime="text/csv",
            )
            
            # Akurasi model
            st.subheader("🎯 Evaluasi Model")
            
            st.write("""
            **Akurasi ARIMA** bergantung pada pola data dan kompleksitas. Beberapa faktor penting:
            
            1. **Panjang Data**: Data lebih panjang umumnya menghasilkan prediksi lebih akurat
            2. **Stasioneritas**: Data yang stasioner cenderung lebih mudah diprediksi
            3. **Musim/Tren**: Pola musiman dan tren mempengaruhi akurasi prediksi
            """)
            
            with st.expander("🔍 Detail Model ARIMA"):
                st.code(str(model_fit.summary()))
        
        except Exception as e:
            st.error(f"Terjadi kesalahan saat pemodelan: {str(e)}")
            st.info("Coba tambahkan lebih banyak data historis atau ubah parameter model untuk hasil yang lebih baik.")
    
    # Tips penggunaan
    with st.expander("💡 Tips Penggunaan Prediksi", expanded=False):
        st.write("""
        **Tips Meningkatkan Kualitas Prediksi:**
        
        1. **Data Historis Lengkap**: Pastikan ada data transaksi keluar yang konsisten untuk minimal 1-2 tahun
        2. **Perhatikan Musiman**: Pertimbangkan pola musiman dalam pengambilan keputusan pengadaan
        3. **Evaluasi Berkala**: Bandingkan prediksi dengan aktual secara berkala untuk meningkatkan akurasi
        4. **Kombinasikan Metode**: Gunakan prediksi sebagai salah satu faktor dalam keputusan persediaan
        """)
    
    conn.close()