import streamlit as st
import sqlite3
from datetime import datetime
import pandas as pd
import time


def main():
    st.header("📤 Transaksi Keluar", divider="green")
    conn = sqlite3.connect('data/stok.db')
    c = conn.cursor()

    # Ambil data produk
    produk = c.execute("SELECT id, nama, stok FROM produk").fetchall()
    if not produk:
        st.warning(
            "Tidak ada produk tersedia. Silakan tambah produk terlebih dahulu.", icon="⚠️")
        return

    # Input Produk dan Jumlah (di luar form untuk responsivitas)
    daftar_produk = [f"{p[1]} (Stok: {p[2]})" for p in produk]
    produk_pilihan = st.selectbox(
        "📦 Pilih Produk",
        daftar_produk,
        format_func=lambda x: x,
        help="Pilih produk yang akan dikeluarkan"
    )

    # Ambil data produk terpilih
    selected_produk = next(
        (p for p in produk if f"{p[1]} (Stok: {p[2]})" == produk_pilihan),
        None
    )
    max_jumlah = selected_produk[2] if selected_produk else 0

    # Input jumlah dengan validasi
    jumlah = st.number_input(
        "Jumlah",
        min_value=1,
        max_value=max_jumlah,
        step=1,
        help="Masukkan jumlah barang keluar"
    )

    # Progress bar dinamis
    if max_jumlah > 0:
        progress = jumlah / max_jumlah
        progress_text = f"{jumlah}/{max_jumlah} tersedia"
        st.progress(progress, text=progress_text)

        # Peringatan stok rendah
        if jumlah > max_jumlah * 0.8:
            st.warning("Stok tersisa kurang dari 20%!", icon="⚠️")
        elif jumlah > max_jumlah:
            st.error("Jumlah melebihi stok!", icon="❌")

    # Form untuk finalisasi transaksi
    with st.form("tambah_keluar_form", border=True):
        st.subheader("📅 Detail Transaksi")
        tanggal = st.date_input(
            "Transaksi",
            value=datetime.now(),
            help="Pilih tanggal transaksi"
        )

        submitted = st.form_submit_button("Tambah Transaksi", type="primary")

        if submitted:
            if not selected_produk:
                st.error("Produk tidak valid!", icon="❌")
                return
            if jumlah > selected_produk[2]:
                st.error(
                    f"Stok {selected_produk[1]} tidak mencukupi!", icon="❌")
                return

            with st.spinner("Menyimpan transaksi..."):
                time.sleep(1)
                try:
                    with conn:
                        c.execute("INSERT INTO transaksi_keluar (produk_id, jumlah, tanggal) VALUES (?, ?, ?)",
                                  (selected_produk[0], jumlah, tanggal))
                        c.execute("UPDATE produk SET stok = stok - ? WHERE id = ?",
                                  (jumlah, selected_produk[0]))
                    st.success('Transaksi berhasil!', icon="✅")
                    st.session_state.page_keluar = 1
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}", icon="❌")

    # Tampilkan riwayat transaksi
    items_per_page = 5
    page_number = st.session_state.get('page_keluar', 1)
    offset = (page_number - 1) * items_per_page

    total_transaksi = c.execute(
        "SELECT COUNT(*) FROM transaksi_keluar").fetchone()[0]
    total_pages = (total_transaksi // items_per_page) + \
        (1 if total_transaksi % items_per_page > 0 else 0)

    # Pagination Controls
    with st.container():
        col_prev, col_info, col_next, col_jump = st.columns([1, 2, 1, 2])
        with col_prev:
            if st.button("⬅️ Sebelumnya", disabled=(page_number == 1), key="prev_keluar"):
                st.session_state.page_keluar = page_number - 1
                st.rerun()
        with col_info:
            st.write(f"Halaman {page_number} dari {total_pages}")
        with col_next:
            if st.button("Selanjutnya ➡️", disabled=(page_number == total_pages), key="next_keluar"):
                st.session_state.page_keluar = page_number + 1
                st.rerun()
        with col_jump:
            new_page = st.number_input(
                "Lompat ke halaman",
                min_value=1,
                max_value=total_pages,
                value=page_number,
                step=1,
                key="jump_keluar"
            )
            if new_page != page_number:
                st.session_state.page_keluar = new_page
                st.rerun()

    # Tampilkan Data Transaksi
    transaksi = c.execute("""
        SELECT 
            tk.id,
            p.nama AS Produk,
            tk.jumlah AS Jumlah,
            strftime('%d-%m-%Y', tk.tanggal) AS Tanggal
        FROM transaksi_keluar tk
        JOIN produk p ON tk.produk_id = p.id
        LIMIT ? OFFSET ?
    """, (items_per_page, offset)).fetchall()

    if not transaksi:
        st.info("Tidak ada riwayat transaksi keluar.", icon="텅")
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
                    help="Jumlah barang keluar"
                ),
                "Tanggal": "Tanggal Transaksi"
            }
        )

    conn.close()
