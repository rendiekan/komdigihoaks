import streamlit as st
import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# --- KONFIGURASI HALAMAN STREAMLIT ---
st.set_page_config(page_title="Analisis Hoaks KOMDIGI", layout="wide")
st.title("Aplikasi Analisis & Klasifikasi Berita Hoaks KOMDIGI")

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
    # Pastikan file dataset utama ada
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

# --- TRAINING & MENG-CACHE MODEL ---
@st.cache_resource
def train_all_models(dataframe):
    X = dataframe['stemmed_body'].fillna('')
    y = dataframe['target_label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    # 1. Logistic Regression
    model_lr = LogisticRegression(max_iter=1000).fit(X_train_tfidf, y_train)
    
    # 2. Decision Tree
    model_dt = DecisionTreeClassifier(max_depth=20, criterion='gini', random_state=42).fit(X_train_tfidf, y_train)
    
    # 3. Random Forest
    model_rf = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42).fit(X_train_tfidf, y_train)
    
    # 4. Naive Bayes
    model_nb = MultinomialNB(alpha=1.0).fit(X_train_tfidf, y_train)
    
    # 5. K-Means Clustering (Unsupervised)
    tfidf_kmeans = TfidfVectorizer(max_features=1000)
    X_tfidf_km = tfidf_kmeans.fit_transform(dataframe['body_text'].fillna(''))
    model_kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, random_state=42).fit(X_tfidf_km)
    
    # Simpan report evaluasi
    reports = {
        "Logistic Regression": (model_lr, accuracy_score(y_test, model_lr.predict(X_test_tfidf)), classification_report(y_test, model_lr.predict(X_test_tfidf))),
        "Decision Tree": (model_dt, accuracy_score(y_test, model_dt.predict(X_test_tfidf)), classification_report(y_test, model_dt.predict(X_test_tfidf))),
        "Random Forest": (model_rf, accuracy_score(y_test, model_rf.predict(X_test_tfidf)), classification_report(y_test, model_rf.predict(X_test_tfidf))),
        "Naive Bayes": (model_nb, accuracy_score(y_test, model_nb.predict(X_test_tfidf)), classification_report(y_test, model_nb.predict(X_test_tfidf)))
    }
    
    return tfidf_vectorizer, reports, tfidf_kmeans, model_kmeans

tfidf_vec, trained_models, tfidf_km, kmeans_model = train_all_models(df)

# --- NAVIGASI ANTARMUKA (TABS) ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Uji Prediksi Berita", 
    "📊 Performa & Evaluasi Model", 
    "📁 Klasterisasi Topik (K-Means)", 
    "📈 Eksplorasi Data"
])

# --- TAB 1: UJI PREDIKSI BERITA ---
with tab1:
    st.header("Deteksi Kategori Berita Hoaks")
    st.write("Masukkan teks berita di bawah ini untuk memprediksi topik kategorinya secara real-time.")
    
    input_text = st.text_area("Input Teks Berita / Artikel:", height=200, placeholder="Tempel teks berita di sini...")
    pilihan_model = st.selectbox("Pilih Model Machine Learning:", list(trained_models.keys()))
    
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
                
                # Ambil model pilihan dan lakukan prediksi
                current_model = trained_models[pilihan_model][0]
                prediksi = current_model.predict(input_tfidf)[0]
                
                st.success(f"### Hasil Prediksi Kategori: **{prediksi}**")
                st.info(f"Dianalisis menggunakan model: *{pilihan_model}*")

# --- TAB 2: PERFORMA & EVALUASI MODEL ---
with tab2:
    st.header("Laporan Performa Model Klasifikasi (Supervised Learning)")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Logistic Regression", f"{trained_models['Logistic Regression'][1]:.3f}")
    col2.metric("Decision Tree", f"{trained_models['Decision Tree'][1]:.3f}")
    col3.metric("Random Forest", f"{trained_models['Random Forest'][1]:.3f}")
    col4.metric("Naive Bayes", f"{trained_models['Naive Bayes'][1]:.3f}")
    
    st.write("---")
    pilih_report = st.selectbox("Lihat Detail Laporan Klasifikasi:", list(trained_models.keys()))
    st.text(f"Classification Report untuk {pilih_report}:")
    st.code(trained_models[pilih_report][2])

# --- TAB 3: KLASTERISASI TOPIK (K-MEANS) ---
with tab3:
    st.header("Hasil K-Means Clustering (Unsupervised Learning)")
    st.write("Mengelompokkan berita hoaks menjadi 5 kelompok tren utama berdasarkan kesamaan kata.")
    
    centroids = kmeans_model.cluster_centers_.argsort()[:, ::-1]
    fitur_kata = tfidf_km.get_feature_names_out()
    df['cluster_label'] = kmeans_model.labels_
    
    for i in range(5):
        with st.container():
            st.subheader(f"Group Cluster {i+1}")
            top_words = [fitur_kata[ind] for ind in centroids[i, :8]]
            st.write(f"**Kata Kunci Dominan:** {', '.join(top_words)}")
            
            jumlah_doc = len(df[df['cluster_label'] == i])
            st.write(f"**Jumlah Anggota Dokumen:** {jumlah_doc} berita")
            st.write("---")

# --- TAB 4: EKSPLORASI DATA ---
with tab4:
    st.header("Sampel Data Teratas")
    st.dataframe(df[['title', 'body_text', 'target_label']].head(10))
    
    st.header("Informasi Dataset Kosong")
    st.write(df.isnull().sum())
