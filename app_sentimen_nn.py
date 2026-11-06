import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Dashboard Analisis Sentimen Neural Network",
    layout="wide"
)

st.title("🔥 Dashboard Analisis Sentimen Kebakaran Hutan")
st.markdown("Aplikasi analisis sentimen berbasis **Neural Network (Multilayer Perceptron)** dan **TF-IDF Vectorizer**.")

# -----------------------------------------------------------------------------
# Fungsi Pemrosesan Teks & Pelabelan
# -----------------------------------------------------------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

positif_words = ['aman', 'lega', 'selamat', 'bantuan', 'pulih', 'padam', 'berhasil', 'terkendali', 'semangat', 'apresiasi', 'dukungan', 'donasi', 'tanggap', 'pahlawan', 'relawan', 'evakuasi', 'baik']
negatif_words = ['banjir', 'parah', 'rusak', 'macet', 'bahaya', 'kebakaran', 'bencana', 'darurat', 'korban', 'asap', 'membara', 'terbakar', 'api', 'krisis', 'ancaman', 'duka', 'mati', 'rugi', 'tewas', 'sesak', 'racun', 'boikot', 'penyebab', 'hukum']

def label_sentiment(text):
    words = text.split()
    pos_count = sum(1 for w in words if w in positif_words)
    neg_count = sum(1 for w in words if w in negatif_words)
    
    if pos_count > neg_count:
        return 'positif'
    elif neg_count > pos_count:
        return 'negatif'
    else:
        return 'netral'

# -----------------------------------------------------------------------------
# Memuat Data
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r'D:\Analisis sentimen menggunakan Neural Network\tweet_data_kebakaran_hutan.csv')
    df['clean_text'] = df['full_text'].apply(clean_text)
    df['sentimen'] = df['clean_text'].apply(label_sentiment)
    return df

try:
    df = load_data()
except Exception as e:
    st.error("Gagal memuat dataset 'tweet_data_kebakaran_hutan.csv'. Pastikan file berada di folder yang sama.")
    st.stop()

# -----------------------------------------------------------------------------
# Train Model
# -----------------------------------------------------------------------------
@st.cache_resource
def train_nn_model(df):
    X = df['clean_text']
    y = df['sentimen']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=300, random_state=42)
    mlp.fit(X_train_tfidf, y_train)
    
    y_pred = mlp.predict(X_test_tfidf)
    return mlp, vectorizer, X_test, y_test, y_pred

mlp_model, vectorizer, X_test, y_test, y_pred = train_nn_model(df)

# -----------------------------------------------------------------------------
# Navigasi & Tampilan
# -----------------------------------------------------------------------------
st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", [
    "Dataset & Cleansing",
    "Distribusi Sentimen",
    "WordCloud",
    "Evaluasi Model NN",
    "Prediksi Teks Baru"
])

if menu == "Dataset & Cleansing":
    st.header("Dataset Hasil Pembersihan & Pelabelan")
    st.dataframe(df[['full_text', 'clean_text', 'sentimen']].head(10), use_container_width=True)
    st.info(f"Total Dataset: {len(df)} baris")

elif menu == "Distribusi Sentimen":
    st.header("Distribusi Sentimen")
    counts = df['sentimen'].value_counts()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Bar Chart")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=counts.index, y=counts.values, palette='viridis', ax=ax)
        st.pyplot(fig)
    with col2:
        st.subheader("Pie Chart")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=140)
        st.pyplot(fig)

elif menu == "WordCloud":
    st.header("WordCloud Per Kategori Sentimen")
    cols = st.columns(3)
    sentimens = ['positif', 'netral', 'negatif']
    cmaps = ['Greens', 'Blues', 'Reds']
    
    for idx, s in enumerate(sentimens):
        with cols[idx]:
            st.subheader(f"Sentimen {s.capitalize()}")
            text = " ".join(df[df['sentimen'] == s]['clean_text'])
            if text.strip():
                wc = WordCloud(width=400, height=300, background_color='white', colormap=cmaps[idx]).generate(text)
                fig, ax = plt.subplots()
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)

elif menu == "Evaluasi Model NN":
    st.header("Evaluasi Model Neural Network (MLP)")
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Akurasi", f"{acc*100:.2f}%")
    c2.metric("Presisi", f"{prec*100:.2f}%")
    c3.metric("Recall", f"{rec*100:.2f}%")
    c4.metric("F1-Score", f"{f1*100:.2f}%")
    
    labels = sorted(list(set(y_test)))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    st.pyplot(fig)

elif menu == "Prediksi Teks Baru":
    st.header("Prediksi Sentimen Teks Baru")
    user_text = st.text_area("Masukkan Kalimat/Tweet:")
    
    if st.button("Prediksi"):
        if user_text.strip():
            cleaned = clean_text(user_text)
            vec = vectorizer.transform([cleaned])
            pred = mlp_model.predict(vec)[0]
            proba = mlp_model.predict_proba(vec)[0]
            
            emoji_map = {'positif': '😊 Positif', 'netral': '😐 Netral', 'negatif': '😡 Negatif'}
            st.success(f"Hasil: {emoji_map.get(pred, pred)}")
            st.write("Teks Bersih:", cleaned)
            
            proba_df = pd.DataFrame({'Kategori': mlp_model.classes_, 'Probabilitas': [f"{p*100:.2f}%" for p in proba]})
            st.dataframe(proba_df)