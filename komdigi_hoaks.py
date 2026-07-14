import streamlit as st
import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# --- KONFIGURASI HALAMAN STREAMLIT ---
st.set_page_config(page_title="Analisis Hoaks KOMDIGI (Random Forest)", layout="wide")
st.title("Aplikasi Klasifikasi Berita Hoaks KOMDIGI — Random Forest")

# --- INISIALISASI SASTRAWI ---
stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
stemmer = StemmerFactory().create_stemmer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    return stopword_remover.remove(text)

# --- MEMPROSES & MENG-CACHE DATASET ---
@st.cache_data
def load_and_preprocess_data():
    if not os.path.exists('komdigi_hoaks.csv'):
        st.error("File 'komdigi_hoaks.csv' tidak ditemukan! Pastikan sudah di-upload ke GitHub.")
        st.stop()
        
    df = pd.read_csv('komdigi_hoaks.csv')
    df = df.dropna(subset=['title', 'body_text'])
    
    # Preprocessing text
    df['clean_title'] = df['title'].apply(clean_text)
    df['clean_body'] = df['body_text'].apply(clean_text)
    df['clean_body'] = df['clean_body'].apply(remove_stopwords)
    df['stemmed_body'] = df['clean_body'].apply(lambda x: stemmer.stem(x))
    
    # Menentukan target kolom secara fleksibel ('topics' atau 'category')
    target_col = 'topics' if 'topics' in df.columns else 'category'
    df['target_label'] = df[target_col]
    
    # Menyimpan ke file baru secara aman (Menghindari OSError)
    os.makedirs('drive/MyDrive', exist_ok=True)
    df_ready = df[['id', 'clean_title', 'stemmed_body', 'target_label']]
    df_ready.to_csv('drive/MyDrive/komdigi_hoaks_clean.csv', index=False)
    
    return df

# Spinner loading saat pertama kali aplikasi dibuka
with st.spinner("Sedang memproses dataset dan menjalankan Sastrawi Stemmer... Mohon tunggu sebentar."):
    df = load_and_preprocess_data()

# --- TRAINING & MENG-CACHE MODEL RANDOM FOREST ---
@st.cache_resource
def train_rf_model(dataframe):
    X = dataframe['stemmed_body'].fillna('')
    y = dataframe['target_label']
    
    # Membagi Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Ekstraksi Fitur dengan TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    # Inisialisasi dan Pelatihan Model Random Forest
    model_rf = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
    model_rf.fit(X_train_tfidf, y_train)
    
    # Evaluasi Performa
    y_pred = model_rf.predict(X_test_tfidf)
    akurasi = accuracy_score(y_test, y_pred)
    laporan = classification_report(y_test, y_pred)
    
    return tfidf_vectorizer, model_rf, akurasi, laporan

tfidf_vec, model_rf, akurasi_rf, laporan_rf = train_rf_model(df)

# --- NAVIGASI ANTARMUKA (TABS) ---
tab1, tab2, tab3 = st.tabs([
    "🔍 Uji Prediksi Berita", 
    "📊 Performa Model (Random Forest)", 
    "📈 Eksplorasi Data"
])

# --- TAB 1: UJI PREDIKSI BERITA ---
with tab1:
    st.header("Deteksi Kategori Berita Hoaks")
    st.write("Masukkan teks berita di bawah ini untuk memprediksi topik kategorinya menggunakan model **Random Forest**.")
    
    input_text = st.text_area("Input Teks Berita / Artikel:", height=200, placeholder="Tempel teks berita di sini...")
    
    if st.button("Analisis Teks"):
        if input_text.strip() == "":
            st.warning("Silakan masukkan teks terlebih dahulu!")
        else:
            with st.spinner("Sedang membersihkan teks dan melakukan klasifikasi..."):
                # Proses teks inputan baru
                clean_input = clean_text(input_text)
                clean_input = remove_stopwords(clean_input)
                stemmed_input = stemmer.stem(clean_input)
                
                # Transformasi ke matriks TF-IDF
                input_tfidf = tfidf_vec.transform([stemmed_input])
                
                # Prediksi menggunakan Random Forest
                prediksi = model_rf.predict(input_tfidf)[0]
                
                st.success(f"### Hasil Prediksi Kategori: **{prediksi}**")

# --- TAB 2: PERFORMA MODEL (RANDOM FOREST) ---
with tab2:
    st.header("Laporan Performa Model Random Forest")
    
    # Menampilkan metrik akurasi utama
    st.metric(label="Akurasi Model Random Forest", value=f"{akurasi_rf:.3f} ({akurasi_rf * 100:.1f}%)")
    
    st.write("---")
    st.subheader("Detail Laporan Klasifikasi (Classification Report)")
    st.text("Berikut adalah rincian presisi, recall, dan f1-score untuk setiap kategori hoaks:")
    st.code(laporan_rf)

# --- TAB 3: EKSPLORASI DATA ---
with tab3:
    st.header("Sampel Data Teratas")
    st.write("Menampilkan 10 baris pertama data asli beserta target labelnya.")
    st.dataframe(df[['title', 'body_text', 'target_label']].head(10))
    
    st.header("Informasi Ringkasan Data Kosong")
    st.write(df.isnull().sum())
