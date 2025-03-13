import streamlit as st
import sqlite3
import pandas as pd


def main():
    st.header("📦 Manajemen Produk", divider="green")
    conn = sqlite3.connect('data/stok.db')
    c = conn.cursor()

    # Pencarian dan Filter
    with st.expander("🔍 Filter & Pencarian", expanded=True):
        col_search, col_filter = st.columns([3, 1])
        with col_search:
            search_query = st.text_input(
                "Cari Produk",
                placeholder="Ketik nama produk...",
                help="Cari produk berdasarkan nama"
            )
        with col_filter:
            distinct_satuan = c.execute(
                "SELECT DISTINCT satuan FROM produk").fetchall()
            satuan_options = [sat[0]
                              for sat in distinct_satuan] if distinct_satuan else []
            filter_satuan = st.multiselect(
                "Filter Satuan",
                options=satuan_options,
                placeholder="Pilih satuan",
                help="Filter berdasarkan satuan produk"
            )

    # Pagination
    items_per_page = 5
    page_number = st.session_state.get('page_produk', 1)
    offset = (page_number - 1) * items_per_page

    # Query dengan pencarian dan filter
    query = "SELECT * FROM produk WHERE 1=1"
    params = []

    if search_query:
        query += " AND nama LIKE ?"
        params.append(f"%{search_query}%")
    if filter_satuan:
        query += f" AND satuan IN ({','.join(['?']*len(filter_satuan))})"
        params.extend(filter_satuan)

    total_produk = c.execute(
        f"SELECT COUNT(*) FROM ({query})", params).fetchone()[0]
    total_pages = (total_produk // items_per_page) + \
        (1 if total_produk % items_per_page > 0 else 0)

    # Pagination Controls
    with st.container():
        col_prev, col_info, col_next, col_jump = st.columns([1, 2, 1, 2])
        with col_prev:
            if st.button("⬅️ Sebelumnya", disabled=(page_number == 1)):
                st.session_state.page_produk = page_number - 1
                st.rerun()
        with col_info:
            st.write(f"Halaman {page_number} dari {total_pages}")
        with col_next:
            if st.button("Selanjutnya ➡️", disabled=(page_number == total_pages)):
                st.session_state.page_produk = page_number + 1
                st.rerun()
        with col_jump:
            new_page = st.number_input(
                "Lompat ke halaman",
                min_value=1,
                max_value=total_pages,
                value=page_number,
                step=1
            )
            if new_page != page_number:
                st.session_state.page_produk = new_page
                st.rerun()

    # Form Tambah Produk
    with st.form("tambah_produk_form", border=True):
        st.subheader("➕ Tambah Produk Baru")
        col1, col2 = st.columns(2)
        with col1:
            nama = st.text_input(
                "Nama Produk",
                placeholder="Contoh: Beras Premium",
                help="Masukkan nama lengkap produk"
            )
            if not nama:
                st.warning("Nama produk wajib diisi!", icon="⚠️")
        with col2:
            satuan = st.text_input(
                "Satuan",
                placeholder="Contoh: Kg",
                help="Masukkan satuan (e.g., Pcs, Liter)"
            )
            if not satuan:
                st.warning("Satuan wajib diisi!", icon="⚠️")
        stok = st.number_input(
            "Stok Awal",
            min_value=0,
            value=0,
            step=1,
            help="Masukkan jumlah stok awal"
        )

        submitted = st.form_submit_button("Tambah Produk", type="primary")

        if submitted:
            if not nama or not satuan:
                st.error("Lengkapi semua field wajib!", icon="❌")
            else:
                try:
                    with conn:
                        c.execute("INSERT INTO produk (nama, stok, satuan) VALUES (?, ?, ?)",
                                  (nama, stok, satuan))
                    st.success('Produk berhasil ditambahkan!', icon="✅")
                    st.session_state.page_produk = 1
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}", icon="❌")

    # Tampilkan Data Produk
    produk = c.execute(query + " LIMIT ? OFFSET ?",
                       tuple(params) + (items_per_page, offset)).fetchall()

    if not produk:
        if search_query or filter_satuan:
            st.info("Tidak ada produk yang sesuai kriteria", icon="🔍")
        else:
            st.info(
                "Tidak ada produk tersedia. Silakan tambah produk terlebih dahulu.", icon="텅")
    else:
        df_produk = pd.DataFrame(
            produk, columns=["ID", "Nama", "Stok", "Satuan"])

        # Styling Tabel
        styled_df = df_produk.style \
            .background_gradient(cmap='Blues', subset=['Stok']) \
            .format({'Stok': '{:,}'}) \
            .set_properties(**{'text-align': 'center'})

        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "ID": "ID",
                "Nama": "Nama Produk",
                "Stok": st.column_config.NumberColumn(
                    "Stok",
                    format="%d",
                    help="Jumlah stok saat ini"
                ),
                "Satuan": "Satuan"
            }
        )

    conn.close()
