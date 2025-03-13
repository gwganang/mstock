import streamlit as st
import sqlite3
from datetime import datetime
import pandas as pd
import time


def main():
    st.header("📥 Transaksi Masuk", divider="green")
    conn = sqlite3.connect('data/stok.db')
    c = conn.cursor()

    # Form Tambah Transaksi
    with st.form("tambah_masuk_form", border=True):
        st.subheader("➕ Tambah Transaksi Masuk")
        col1, col2 = st.columns([3, 1])
        with col1:
            produk = c.execute("SELECT id, nama FROM produk").fetchall()
            if not produk:
                st.warning(
                    "Tidak ada produk tersedia. Silakan tambah produk terlebih dahulu.",
                    icon="⚠️"
                )
                st.form_submit_button("Tambah Transaksi", disabled=True)
                return
            daftar_produk = [p[1] for p in produk]
            produk_pilihan = st.selectbox(
                "Pilih Produk",
                daftar_produk,
                format_func=lambda x: f"📦 {x}",
                help="Pilih produk yang akan ditambahkan"
            )
        with col2:
            jumlah = st.number_input(
                "Jumlah",
                min_value=1,
                step=1,
                help="Masukkan jumlah barang masuk"
            )
        tanggal = st.date_input(
            "Tanggal Transaksi",
            value=datetime.now(),
            help="Pilih tanggal transaksi"
        )

        submitted = st.form_submit_button("Tambah Transaksi", type="primary")

        if submitted:
            with st.spinner("Menyimpan transaksi..."):
                time.sleep(1)  # Simulasi loading
                produk_id = [p[0] for p in produk if p[1] == produk_pilihan][0]
                try:
                    with conn:
                        c.execute("INSERT INTO transaksi_masuk (produk_id, jumlah, tanggal) VALUES (?, ?, ?)",
                                  (produk_id, jumlah, tanggal))
                        c.execute("UPDATE produk SET stok = stok + ? WHERE id = ?",
                                  (jumlah, produk_id))
                    st.success('Transaksi berhasil!', icon="✅")
                    st.session_state.page_masuk = 1
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}", icon="❌")

    # Pagination
    items_per_page = 5
    page_number = st.session_state.get('page_masuk', 1)
    offset = (page_number - 1) * items_per_page

    # Hitung total data
    total_transaksi = c.execute(
        "SELECT COUNT(*) FROM transaksi_masuk").fetchone()[0]
    total_pages = (total_transaksi // items_per_page) + \
        (1 if total_transaksi % items_per_page > 0 else 0)

    # Pagination Controls
    with st.container():
        col_prev, col_info, col_next, col_jump = st.columns([1, 2, 1, 2])
        with col_prev:
            if st.button("⬅️ Sebelumnya", disabled=(page_number == 1), key="prev_masuk"):
                st.session_state.page_masuk = page_number - 1
                st.rerun()
        with col_info:
            st.write(f"Halaman {page_number} dari {total_pages}")
        with col_next:
            if st.button("Selanjutnya ➡️", disabled=(page_number == total_pages), key="next_masuk"):
                st.session_state.page_masuk = page_number + 1
                st.rerun()
        with col_jump:
            new_page = st.number_input(
                "Lompat ke halaman",
                min_value=1,
                max_value=total_pages,
                value=page_number,
                step=1,
                key="jump_masuk"
            )
            if new_page != page_number:
                st.session_state.page_masuk = new_page
                st.rerun()

    # Tampilkan Data Transaksi
    transaksi = c.execute("""
        SELECT 
            tm.id,
            p.nama AS Produk,
            tm.jumlah AS Jumlah,
            strftime('%d-%m-%Y', tm.tanggal) AS Tanggal
        FROM transaksi_masuk tm
        JOIN produk p ON tm.produk_id = p.id
        LIMIT ? OFFSET ?
    """, (items_per_page, offset)).fetchall()

    if not transaksi:
        st.info("Tidak ada riwayat transaksi masuk.", icon="텅")
    else:
        df_transaksi = pd.DataFrame(
            transaksi, columns=["ID", "Produk", "Jumlah", "Tanggal"])

        # Styling Tabel
        styled_df = df_transaksi.style \
            .format({'Jumlah': '{:,}'}) \
            .set_properties(**{'text-align': 'center'})

        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "ID": "ID",
                "Produk": "Nama Produk",
                "Jumlah": st.column_config.NumberColumn(
                    "Jumlah",
                    format="%d",
                    help="Jumlah barang masuk"
                ),
                "Tanggal": "Tanggal Transaksi"
            }
        )

    conn.close()
