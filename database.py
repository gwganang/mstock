import sqlite3
import os

def init_db():
    # Pastikan folder data ada
    if not os.path.exists("data"):
        os.makedirs("data")
    
    # Koneksi database
    conn = sqlite3.connect('data/stok.db')
    c = conn.cursor()
    
    # Tabel Produk
    c.execute('''CREATE TABLE IF NOT EXISTS produk (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nama TEXT NOT NULL,
                    stok INTEGER NOT NULL,
                    satuan TEXT NOT NULL)''')
    
    # Tabel Transaksi Masuk
    c.execute('''CREATE TABLE IF NOT EXISTS transaksi_masuk (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    produk_id INTEGER,
                    jumlah INTEGER NOT NULL,
                    tanggal TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(produk_id) REFERENCES produk(id))''')
    
    # Tabel Transaksi Keluar
    c.execute('''CREATE TABLE IF NOT EXISTS transaksi_keluar (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    produk_id INTEGER,
                    jumlah INTEGER NOT NULL,
                    tanggal TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(produk_id) REFERENCES produk(id))''')
    
    conn.commit()
    conn.close()