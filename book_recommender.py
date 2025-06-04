import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_data():
    books = pd.read_csv("books.csv", on_bad_lines='skip')
    books['genres'] = books.get('genres', '')  
    books['text'] = books['title'].fillna('') + " by " + books['authors'].fillna('') + " " + books['genres'].fillna('')
    # Pastikan kolom rating ada, jika tidak, isi dengan 0
    for i in range(1,6):
        col = f'ratings_{i}'
        if col not in books.columns:
            books[col] = 0
    # Jika ada kolom image_url, kalau tidak, buat kosong
    if 'image_url' not in books.columns:
        books['image_url'] = ""
    return books

@st.cache_data
def generate_embeddings(texts):
    model = load_model()
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

def recommend(user_input, books, embeddings):
    model = load_model()
    user_emb = model.encode([user_input])
    similarities = cosine_similarity(user_emb, embeddings)[0]
    top_idx = np.argsort(similarities)[::-1][:5]
    top_books = books.iloc[top_idx].copy()
    top_books['similarity'] = similarities[top_idx]
    return top_books

def rating_stars(row):
    # Buat rating rata-rata dari rating_1..rating_5
    total_ratings = sum(row[f'ratings_{i}'] * i for i in range(1,6))
    count_ratings = sum(row[f'ratings_{i}'] for i in range(1,6))
    if count_ratings == 0:
        return "No ratings"
    avg_rating = total_ratings / count_ratings
    full_stars = int(avg_rating)
    half_star = avg_rating - full_stars >= 0.5
    stars = "⭐" * full_stars
    if half_star:
        stars += "✰"
    return f"{stars} ({avg_rating:.2f}/5 from {count_ratings} ratings)"

st.set_page_config(page_title="📚 Book Recommender System", layout="wide")

st.title("📚 Book Recommender System")
st.markdown("""
Masukkan deskripsi singkat tentang jenis buku yang kamu suka, misalnya:  
*“I want a romantic and funny book with a lighthearted story.”*  
Sistem ini akan merekomendasikan buku yang paling sesuai dengan preferensimu.
""")

books = load_data()
embeddings = generate_embeddings(books['text'].tolist())

user_input = st.text_input("Describe your favorite type of book:")

if st.button("Recommend"):
    if not user_input.strip():
        st.warning("Please enter your book preference first.")
    else:
        with st.spinner("Finding recommendations..."):
            results = recommend(user_input, books, embeddings)
            if results.empty:
                st.write("No recommendations found.")
            else:
                st.subheader("✨ Recommended Books:")
                for _, row in results.iterrows():
                    cols = st.columns([1,3])
                    with cols[0]:
                        if row['image_url']:
                            st.image(row['image_url'], width=120)
                        else:
                            st.text("No image")
                    with cols[1]:
                        st.markdown(f"### {row['title']}")
                        st.markdown(f"*by {row['authors']}*")
                        st.markdown(f"**Genres/Tags:** {row['genres']}")
                        st.markdown(f"**Rating:** {rating_stars(row)}")
                        st.markdown(f"*Similarity:* {row['similarity']:.4f}")
                        st.write("---")
