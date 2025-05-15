import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

# Load and clean the dataset
df = pd.read_csv("books.csv")
df.dropna(subset=["title", "language_code", "authors", "books_count"], inplace=True)

# Prepare TF-IDF for content-based recommendations
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['title'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
df['title_lower'] = df['title'].str.lower()

# Streamlit setup
st.set_page_config(page_title="ChatBook", layout="centered")

st.title("📚 ChatBook — Your Personal Book Recommender")

# Tabs for two options
tab1, tab2 = st.tabs(["🎯 Mood-Based Search", "🤖 Similar Book Search"])

# === TAB 1: Mood-Based ===
with tab1:
    st.subheader("🎯 Find books based on genre, author, and more")

    genre = st.text_input("Genre / keyword (e.g. romance, thriller):")
    author = st.text_input("Author (optional):")
    max_pages = st.number_input("Max number of pages:", min_value=50, max_value=2000, value=300)
    language = st.selectbox("Language preference:", ['en', 'es', 'fr', 'ta', 'de'])

    if st.button("📖 Recommend me books!", key="mood_btn"):
        filtered = df[df['books_count'] <= max_pages]
        filtered = filtered[filtered['language_code'].str.lower().str.startswith(language.lower())]

        if genre:
            filtered = filtered[filtered['title'].str.contains(genre, case=False, na=False)]
        if author:
            filtered = filtered[filtered['authors'].str.contains(author, case=False, na=False)]

        results = filtered[['title', 'authors']].drop_duplicates().head(5).values.tolist()

        if results:
            st.success("Here are some books you might like:")
            for title, writer in results:
                st.markdown(f"📘 **{title}**  \n👤 _by {writer}_")
        else:
            st.warning("😕 No matching books found. Try different filters!")

# === TAB 2: Content-Based AI ===
with tab2:
    st.subheader("🤖 Find books similar to one you liked")

    book_title = st.text_input("Enter a book title you liked:")

    if st.button("🔍 Find Similar Books", key="ai_btn"):
        title = book_title.lower()

        matches = get_close_matches(title, df['title_lower'], n=1, cutoff=0.6)
        if not matches:
            st.warning("Sorry, I couldn't find that book. Try a different title.")
        else:
            matched_title = matches[0]
            idx = df[df['title_lower'] == matched_title].index[0]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
            book_indices = [i[0] for i in sim_scores]
            similar_books = df['title'].iloc[book_indices].tolist()

            st.success("Books similar to your favorite:")
            for b in similar_books:
                st.markdown(f"📗 {b}")
