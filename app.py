import streamlit as st
import pickle
import re
import nltk
import pandas as pd
import os
import requests
import matplotlib.pyplot as plt

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Fake News Detector Pro",
    page_icon="📰",
    layout="wide"
)

# ---------------- LOGIN ----------------

USER = "admin"
PASS = "admin"

if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:
    st.title("🔐 Login")

    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if u == USER and p == PASS:
            st.session_state.login = True
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.stop()

# ---------------- THEME ----------------

if "theme" not in st.session_state:
    st.session_state.theme = "light"

if st.sidebar.button("🌙 Toggle Theme"):
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

bg = "#0f172a" if st.session_state.theme == "dark" else "#f8fafc"
text = "#ffffff" if st.session_state.theme == "dark" else "#111827"

st.markdown(f"""
<style>

.stApp {{
    background: {bg};
    color: {text};
}}

h1,h2,h3,h4,h5,h6 {{
    color: {text} !important;
}}

p, span, label {{
    color: {text} !important;
}}

.stTextInput input, .stTextArea textarea {{
    color: {text} !important;
    background-color: rgba(255, 160, 0, 0.25) !important;
    border: 1px solid rgba(255, 160, 0, 0.25) !important;
}}

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {{
    color: rgba(255, 160, 0, 0.5) !important;
}}

section[data-testid="stSidebar"] {{
    background-color: {bg};
}}

section[data-testid="stSidebar"] * {{
    color: {text} !important;
}}

.stButton>button {{
    background: linear-gradient(90deg,#a5b4fc,#93c5fd);
    color: black;
    border-radius: 12px;
    font-weight: bold;
}}


</style>
""", unsafe_allow_html=True)

# ---------------- NLP ----------------

nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()

# ---------------- LOAD MODEL ----------------

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ---------------- CLEAN ----------------

def clean_text(text):
    text = str(text)
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    return " ".join([stemmer.stem(w) for w in text.split() if w not in STOPWORDS])

# ---------------- SCRAPE ----------------

def extract_article(url):
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        art = soup.find("article")
        p = art.find_all("p") if art else soup.find_all("p")

        return " ".join(x.get_text().strip() for x in p if len(x.get_text()) > 40)

    except:
        return ""

# ---------------- HISTORY ----------------

FILE = "history.csv"

try:
    if os.path.exists(FILE) and os.path.getsize(FILE) > 0:
        history = pd.read_csv(FILE).to_dict("records")
    else:
        history = []
except:
    history = []

# ---------------- UI ----------------

st.title("📰 Fake News Detection Pro")

col1, col2 = st.columns([2, 1])

with col1:
    url = st.text_input("News URL")
    news = st.text_area("Paste News", height=200)
    btn = st.button("Analyze")

with col2:
    st.subheader("Controls")

    if st.button("🗑 Clear History"):
        history = []
        pd.DataFrame(history).to_csv(FILE, index=False)
        st.success("History Cleared")

# ---------------- ANALYSIS ----------------

if btn:

    if url.strip():
        news = extract_article(url)

    if not news.strip():
        st.warning("Enter valid news")
        st.stop()

    vec = vectorizer.transform([clean_text(news)])
    prediction = model.predict(vec)

    score = model.decision_function(vec)
    confidence = round(50 + min(abs(score[0]) * 10, 49), 2)

    if prediction[0] == 0:
        st.error("Fake News")
        result = "Fake News"
    else:
        st.success("Real News")
        result = "Real News"

    st.metric("Confidence", f"{confidence}%")

    history.append({
        "News": news[:100],
        "Result": result,
        "Confidence": confidence
    })

    pd.DataFrame(history).to_csv(FILE, index=False)

# ---------------- DASHBOARD ----------------

st.divider()
st.subheader("📊 Dashboard")

if len(history) > 0:

    df = pd.DataFrame(history)

    col1, col2 = st.columns(2)

    # ---------------- PIE CHART (NO DUPLICATES) ----------------
    with col1:
        st.markdown("### Distribution")

        counts = df["Result"].value_counts()

        labels = ["Fake News", "Real News"]

        values = [
            counts.get("Fake News", 0),
            counts.get("Real News", 0)
        ]

        colors = ["#ef4444", "#22c55e"]

        fig1, ax1 = plt.subplots(figsize=(3.5, 3.5))

        ax1.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            colors=colors
        )

        st.pyplot(fig1)

    # ---------------- BAR CHART (FIXED) ----------------
    with col2:
        st.markdown("### Count View")

        counts = df["Result"].value_counts()

        labels = ["Fake News", "Real News"]

        values = [
            counts.get("Fake News", 0),
            counts.get("Real News", 0)
        ]

        fig2, ax2 = plt.subplots(figsize=(3.5, 3.5))

        ax2.bar(labels, values, color=["#ef4444", "#22c55e"])

        ax2.set_ylabel("Count")
        ax2.set_title("News Count")

        st.pyplot(fig2)

    # ---------------- TABLE ----------------
    st.dataframe(df, use_container_width=True)

else:
    st.info("No data yet")