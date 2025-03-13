import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
from datetime import datetime


def main():
    st.header("📊 Dashboard Stok", divider="green")
    conn = sqlite3.connect('data/stok.db')
    c = conn.cursor()

    # Statistik Utama
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            produk = c.execute("SELECT COUNT(*) FROM produk").fetchone()[0]
            st.metric(
                "📦 Total Produk",
                produk,
                help="Total produk terdaftar",
                delta_color="off"
            )
        with col2:
            stok_total = c.execute(
                "SELECT SUM(stok) FROM produk").fetchone()[0] or 0
            st.metric(
                "📈 Total Stok",
                f"{stok_total:,}",
                help="Total stok keseluruhan",
                delta_color="off"
            )
        with col3:
            transaksi_masuk = c.execute(
                "SELECT COUNT(*) FROM transaksi_masuk").fetchone()[0]
            st.metric(
                "📥 Transaksi Masuk",
                transaksi_masuk,
                help="Total transaksi masuk",
                delta_color="off"
            )
        with col4:
            transaksi_keluar = c.execute(
                "SELECT COUNT(*) FROM transaksi_keluar").fetchone()[0]
            st.metric(
                "📤 Transaksi Keluar",
                transaksi_keluar,
                help="Total transaksi keluar",
                delta_color="off"
            )

    # Grafik Stok Produk
    st.subheader("📌 Stok Produk Terakhir")
    df_produk = pd.read_sql_query("SELECT nama, stok FROM produk", conn)

    fig = px.bar(
        df_produk,
        x='nama', y='stok',
        title='Stok Produk Terakhir',
        labels={'nama': 'Produk', 'stok': 'Jumlah Stok'},
        template='plotly_white',
        hover_data={'nama': True, 'stok': ':,'}
    )
    fig.update_traces(marker_color='#2ECC71')
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title=None,
        yaxis_title=None
    )
    st.plotly_chart(fig, use_container_width=True)

    # Riwayat Transaksi
    st.subheader("📚 Riwayat Transaksi")
    col_masuk, col_keluar = st.columns(2)

    with col_masuk:
        st.write("5 Transaksi Masuk Terakhir")
        df_masuk = pd.read_sql_query('''
            SELECT 
                p.nama AS Produk, 
                tm.jumlah AS Jumlah, 
                strftime('%d-%m-%Y', tm.tanggal) AS Tanggal 
            FROM transaksi_masuk tm
            JOIN produk p ON tm.produk_id = p.id
            ORDER BY tm.tanggal DESC
            LIMIT 5
        ''', conn)
        st.dataframe(
            df_masuk.style.format({'Jumlah': '{:,}'}),
            use_container_width=True,
            hide_index=True
        )

    with col_keluar:
        st.write("5 Transaksi Keluar Terakhir")
        df_keluar = pd.read_sql_query('''
            SELECT 
                p.nama AS Produk, 
                tk.jumlah AS Jumlah, 
                strftime('%d-%m-%Y', tk.tanggal) AS Tanggal 
            FROM transaksi_keluar tk
            JOIN produk p ON tk.produk_id = p.id
            ORDER BY tk.tanggal DESC
            LIMIT 5
        ''', conn)
        st.dataframe(
            df_keluar.style.format({'Jumlah': '{:,}'}),
            use_container_width=True,
            hide_index=True
        )

    conn.close()
