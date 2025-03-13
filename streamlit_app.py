import streamlit as st
from database import init_db
import os

# Konfigurasi Awal
st.set_page_config(
    page_title="MStock - Sistem Manajemen Stok",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': 'https://github.com/your-repo/issues',
    }
)

# Inisialisasi Database
init_db()

# CSS Customization
st.markdown(
    """
    <style>
    /* Global Styles */
    .stApp {
        padding-top: 2rem;
    }
    .stButton>button {
        background-color: #2ECC71;
        color: white;
        border-radius: 8px;
        padding: 0.8rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #28B463;
        transform: translateY(-1px);
    }
    .stAlert {
        border-radius: 8px;
    }
    .sidebar .sidebar-content {
        background-color: #F8F9FA;
    }
    /* Tooltip Styling */
    .tooltip-content {
        font-size: 0.8rem;
        color: #666;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar Navigation
logo_path = "icon/icon.png"
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, use_container_width=True, caption="MStock Inventory")
else:
    st.sidebar.image("https://via.placeholder.com/150", use_container_width=True, caption="MStock Inventory")

pages = {
    "📊 Dashboard": "Dashboard",
    "📦 Produk": "Produk",
    "📥 Transaksi Masuk": "Transaksi Masuk",
    "📤 Transaksi Keluar": "Transaksi Keluar",
    "📈 Prediksi Stok": "Prediksi Stok"  # Menambahkan menu prediksi
}

selected_page = st.sidebar.selectbox(
    "Navigasi",
    options=list(pages.keys()),
    index=0,
    key="navigation",
    help="Pilih halaman yang ingin ditampilkan"
)
page = pages[selected_page]

# Breadcrumb
st.sidebar.markdown(f"**Aktif:** {selected_page}")

# Routing
if page == "Dashboard":
    from dashboard import main as dashboard_page
    dashboard_page()
elif page == "Produk":
    from produk import main as produk_page
    produk_page()
elif page == "Transaksi Masuk":
    from transaksi_masuk import main as masuk_page
    masuk_page()
elif page == "Prediksi Stok":  # Routing untuk menu prediksi
    from prediksi import main as prediksi_page
    prediksi_page()
else:  # Transaksi Keluar
    from transaksi_keluar import main as keluar_page
    keluar_page()