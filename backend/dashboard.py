import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import os

from collections import Counter
from wordcloud import WordCloud

import re
import ast

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#from gensim import corpora
#from gensim.models import LdaModel

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="TextLens AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main{
    background:#0E1117;
}

section[data-testid="stSidebar"]{
    background:#111827;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
    padding-left:3rem;
    padding-right:3rem;
}

.big-title{
    font-size:42px;
    font-weight:700;
    color:white;
}

.sub-title{
    color:#B9BDC7;
    font-size:18px;
}

.card{

    background:#1A2233;

    border-radius:14px;

    padding:18px;

    border:1px solid #2D3748;

}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD DATA
# ==========================================================



@st.cache_data
def load_default_dataset():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(
        BASE_DIR,
        "data",
        "reddit_posts.csv"
    )

    return pd.read_csv(data_path)


uploaded_file = st.sidebar.file_uploader(

    "📂 Upload CSV",

    type=["csv"]

)

if uploaded_file is None:

    df = load_default_dataset()

    st.sidebar.success(
        "Default Dataset Loaded"
    )

else:

    df = pd.read_csv(uploaded_file)

    st.sidebar.success(
        "Custom Dataset Loaded"
    )

# ==========================================================
# TEXT COLUMN DETECTION
# ==========================================================

text_columns = list(

    df.select_dtypes(
        include="object"
    ).columns

)

if len(text_columns)==0:

    st.error(
        "No text columns found in dataset."
    )

    st.stop()

selected_column = st.sidebar.selectbox(

    "📝 Text Column",

    text_columns

)

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.markdown("---")

st.sidebar.title("🧠 TextLens AI")

page = st.sidebar.radio(

    "Navigation",

    [

        "🏠 Overview",

        "🤖 AI Playground",

        "📂 Dataset",

        "☁️ Word Cloud",

        "📈 Word Frequency",

        "😊 Sentiment",

        "🧠 Topic Modeling",

        "📝 Summarizer"

    ]

)

st.sidebar.markdown("---")

st.sidebar.info("""

Current Version

✅ Upload CSV

✅ AI Playground

✅ Analytics

✅ NLP

""")

# ==========================================================
# OVERVIEW PAGE
# ==========================================================

if page == "🏠 Overview":

    st.markdown(
        """
        <div class='big-title'>
        🧠 TextLens AI
        </div>

        <div class='sub-title'>
        AI-Powered Text Analytics Platform
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("Analyze • Visualize • Summarize • Discover Insights")

    st.markdown("---")

    total_rows = len(df)

    total_columns = len(df.columns)

    total_missing = int(df.isnull().sum().sum())

    avg_words = round(
        df[selected_column]
        .astype(str)
        .str.split()
        .str.len()
        .mean(),
        2
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "📄 Total Records",
            f"{total_rows:,}"
        )

    with c2:
        st.metric(
            "📑 Columns",
            total_columns
        )

    with c3:
        st.metric(
            "⚠ Missing Values",
            total_missing
        )

    with c4:
        st.metric(
            "📝 Avg Words",
            avg_words
        )

    st.markdown("---")

    left, right = st.columns([2,1])

    with left:

        st.subheader("📌 About TextLens AI")

        st.write(
        """
TextLens AI is a Natural Language Processing platform that analyzes
any text dataset.

Current Features:

• Upload Any CSV

• Dataset Explorer

• AI Playground

• Word Cloud

• Word Frequency

• Sentiment Analysis

• Topic Modeling

• Text Summarization

Supported datasets:

✔ Reddit

✔ Twitter

✔ YouTube

✔ Amazon Reviews

✔ News Articles

✔ Custom CSV
"""
        )

    with right:

        st.subheader("🚀 Tech Stack")

        st.code(
"""
Python

Pandas

Streamlit

Plotly

spaCy

NLTK

Gensim

WordCloud

VADER

Sumy
"""
        )

    st.markdown("---")

    st.subheader("📊 Dataset Preview")

    st.dataframe(
        df.head(10),
        width="stretch"
    )

    st.markdown("---")

    st.subheader("📈 Dataset Statistics")

   
    stats = pd.DataFrame({

    "Statistic":[
        "Rows",
        "Columns",
        "Missing Values",
        "Text Column"
    ],

    "Value":[
        str(total_rows),
        str(total_columns),
        str(total_missing),
        str(selected_column)
    ]

})

    st.dataframe(
        stats,
        width="stretch"
    )

    st.success(
        "🎉 Upload another CSV anytime from the sidebar."
    )

# ==========================================================
# AI PLAYGROUND
# ==========================================================

elif page == "🤖 AI Playground":

    st.title("🤖 AI Playground")

    st.caption(
        "Analyze any text using Natural Language Processing."
    )

    text = st.text_area(

        "Paste your text here",

        height=220,

        placeholder="Example: Artificial Intelligence is transforming healthcare..."

    )

    if st.button("🚀 Analyze Text"):

        if text.strip() == "":

            st.warning("Please enter some text.")

        else:

            analyzer = SentimentIntensityAnalyzer()

            score = analyzer.polarity_scores(text)

            compound = score["compound"]

            if compound >= 0.05:

                sentiment = "😊 Positive"

            elif compound <= -0.05:

                sentiment = "😡 Negative"

            else:

                sentiment = "😐 Neutral"

            words = re.findall(

                r"\b[a-zA-Z]{3,}\b",

                text.lower()

            )

            total_words = len(text.split())

            total_characters = len(text)

            unique_words = len(set(words))

            reading_time = round(

                total_words / 200,

                2

            )

            top_words = Counter(words).most_common(10)

            c1,c2,c3,c4 = st.columns(4)

            c1.metric(

                "😊 Sentiment",

                sentiment

            )

            c2.metric(

                "📝 Words",

                total_words

            )

            c3.metric(

                "🔤 Characters",

                total_characters

            )

            c4.metric(

                "📚 Reading Time",

                f"{reading_time} min"

            )

            st.markdown("---")

            left,right = st.columns([1,2])

            with left:

                st.subheader("🔑 Top Keywords")

                keyword_df = pd.DataFrame(

                    top_words,

                    columns=[

                        "Keyword",

                        "Frequency"

                    ]

                )

                st.dataframe(

                    keyword_df,

                    width="stretch"

                )

            with right:

                st.subheader("📊 Keyword Frequency")

                fig = px.bar(

                    keyword_df,

                    x="Keyword",

                    y="Frequency",

                    color="Frequency",

                    text="Frequency"

                )

                fig.update_layout(

                    template="plotly_dark",

                    height=450

                )

                st.plotly_chart(

                    fig,

                    width="stretch"

                )

            st.markdown("---")

            st.subheader("📝 Original Text")

            st.write(text)

            st.markdown("---")

            # ==========================================
            # TEXT SUMMARY
            # ==========================================

            st.subheader("📝 AI Summary")

            try:

                parser = PlaintextParser.from_string(

                    text,

                    Tokenizer("english")

                )

                summarizer = LsaSummarizer()

                summary = summarizer(

                    parser.document,

                    2

                )

                summary_text = " ".join(

                    str(sentence)

                    for sentence in summary

                )

                if summary_text.strip()=="":

                    summary_text="Summary could not be generated."

                st.success(summary_text)

            except:

                st.warning("Unable to generate summary.")

            st.markdown("---")

            # ==========================================
            # TOP WORDS TABLE
            # ==========================================

            st.subheader("📋 Top 10 Words")

            freq_df = pd.DataFrame(

                top_words,

                columns=[

                    "Word",

                    "Frequency"

                ]

            )

            st.dataframe(

                freq_df,

                use_container_width=True

            )

            st.markdown("---")

            # ==========================================
            # QUICK INSIGHTS
            # ==========================================

            st.subheader("📌 Quick Insights")

            longest = max(

                words,

                key=len,

                default="-"

            )

            shortest = min(

                words,

                key=len,

                default="-"

            )

            col1,col2,col3 = st.columns(3)

            col1.metric(

                "Longest Word",

                longest

            )

            col2.metric(

                "Shortest Word",

                shortest

            )

            col3.metric(

                "Unique Words",

                unique_words
            )

            st.markdown("---")

            st.download_button(

                "⬇ Download Summary",

                summary_text,

                file_name="summary.txt"

            )

# ==========================================================
# DATASET EXPLORER
# ==========================================================

elif page == "📂 Dataset":

    st.title("📂 Dataset Explorer")

    st.caption(
        "Explore any uploaded CSV dataset."
    )

    search = st.text_input(
        "🔍 Search Dataset"
    )

    display_df = df.copy()

    if search:

        mask = display_df.astype(str).apply(

            lambda row: row.str.contains(

                search,

                case=False,

                na=False

            )

        ).any(axis=1)

        display_df = display_df[mask]

    c1,c2,c3 = st.columns(3)

    c1.metric(

        "Rows",

        len(display_df)

    )

    c2.metric(

        "Columns",

        len(display_df.columns)

    )

    c3.metric(

        "Missing",

        int(display_df.isnull().sum().sum())

    )

    st.markdown("---")

    st.dataframe(

        display_df,

        use_container_width=True,

        height=500

    )

    st.markdown("---")

    st.subheader("📋 Dataset Information")

    info = pd.DataFrame({

        "Column":display_df.columns,

        "Datatype":[

            str(i)

            for i in display_df.dtypes

        ]

    })

    st.dataframe(

        info,

        use_container_width=True

    )

    st.download_button(

        "⬇ Download Dataset",

        display_df.to_csv(index=False),

        "dataset.csv"

    )


# ==========================================================
# WORD CLOUD
# ==========================================================

elif page == "☁️ Word Cloud":

    st.title("☁️ Word Cloud")

    st.caption(
        "Automatically generated from selected text column."
    )

    text = " ".join(

        df[selected_column]

        .astype(str)

    )

    wc = WordCloud(

        width=1600,

        height=700,

        background_color="white",

        colormap="viridis",

        max_words=300

    ).generate(text)

    fig,ax = plt.subplots(

        figsize=(16,8)

    )

    ax.imshow(wc)

    ax.axis("off")

    st.pyplot(fig)

    st.markdown("---")

    st.success(

        f"Generated using '{selected_column}'"

    )

# ==========================================================
# WORD FREQUENCY
# ==========================================================

elif page == "📈 Word Frequency":

    st.title("📈 Word Frequency Analysis")

    st.caption(
        "Most frequently occurring words in the selected text column."
    )

    text = " ".join(

        df[selected_column]

        .astype(str)

        .str.lower()

    )

    words = re.findall(

        r"\b[a-zA-Z]{3,}\b",

        text

    )

    freq = Counter(words)

    freq_df = pd.DataFrame(

        freq.items(),

        columns=[

            "Word",

            "Frequency"

        ]

    )

    freq_df = freq_df.sort_values(

        "Frequency",

        ascending=False

    )

    top_n = st.slider(

        "Select Top Words",

        10,

        100,

        20

    )

    display_df = freq_df.head(top_n)

    st.dataframe(

        display_df,

        use_container_width=True,

        height=400

    )

    st.markdown("---")

    col1,col2 = st.columns(2)

    with col1:

        fig = px.bar(

            display_df,

            x="Word",

            y="Frequency",

            color="Frequency",

            text="Frequency",

            title="Top Words"

        )

        fig.update_layout(

            template="plotly_dark",

            height=500

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    with col2:

        fig2 = px.pie(

            display_df,

            names="Word",

            values="Frequency",

            hole=0.45,

            title="Word Distribution"

        )

        fig2.update_layout(

            template="plotly_dark",

            height=500

        )

        st.plotly_chart(

            fig2,

            use_container_width=True

        )

    st.markdown("---")

    st.subheader("📌 Quick Statistics")

    c1,c2,c3 = st.columns(3)

    c1.metric(

        "Unique Words",

        len(freq_df)

    )

    c2.metric(

        "Top Word",

        display_df.iloc[0]["Word"]

    )

    c3.metric(

        "Frequency",

        int(display_df.iloc[0]["Frequency"])

    )

    st.download_button(

        "⬇ Download Word Frequency",

        display_df.to_csv(index=False),

        file_name="word_frequency.csv",

        mime="text/csv"

    ) 

# ==========================================================
# SENTIMENT ANALYSIS
# ==========================================================

elif page == "😊 Sentiment":

    st.title("😊 Sentiment Analysis")

    st.caption(
        "Analyze the sentiment of the selected text column."
    )

    analyzer = SentimentIntensityAnalyzer()

    sentiments = []

    progress = st.progress(0)

    total = len(df)

    for i, text in enumerate(df[selected_column].astype(str)):

        score = analyzer.polarity_scores(text)

        compound = score["compound"]

        if compound >= 0.05:

            sentiments.append("Positive")

        elif compound <= -0.05:

            sentiments.append("Negative")

        else:

            sentiments.append("Neutral")

        progress.progress((i + 1) / total)

    sentiment_df = df.copy()

    sentiment_df["Sentiment"] = sentiments

    counts = sentiment_df["Sentiment"].value_counts()

    c1, c2, c3 = st.columns(3)

    c1.metric(

        "😊 Positive",

        int(counts.get("Positive", 0))

    )

    c2.metric(

        "😐 Neutral",

        int(counts.get("Neutral", 0))

    )

    c3.metric(

        "😡 Negative",

        int(counts.get("Negative", 0))

    )

    st.markdown("---")

    left, right = st.columns(2)

    with left:

        chart_df = counts.reset_index()

        chart_df.columns = [

            "Sentiment",

            "Count"

        ]

        fig = px.bar(

            chart_df,

            x="Sentiment",

            y="Count",

            color="Sentiment",

            text="Count",

            title="Sentiment Distribution"

        )

        fig.update_layout(

            template="plotly_dark",

            height=500

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    with right:

        fig2 = px.pie(

            chart_df,

            names="Sentiment",

            values="Count",

            hole=0.45,

            title="Sentiment Ratio"

        )

        fig2.update_layout(

            template="plotly_dark",

            height=500

        )

        st.plotly_chart(

            fig2,

            use_container_width=True

        )

    st.markdown("---")

    st.subheader("📋 Dataset with Sentiment")

    st.dataframe(

        sentiment_df,

        use_container_width=True,

        height=400

    )

    st.download_button(

        "⬇ Download Sentiment Dataset",

        sentiment_df.to_csv(index=False),

        file_name="sentiment_results.csv",

        mime="text/csv"

    )

# ==========================================================
# TOPIC MODELING
# ==========================================================

elif page == "🧠 Topic Modeling":

    st.title("🧠 Topic Modeling")

    st.caption(
        "Discover hidden topics from your uploaded dataset."
    )

    sample_df = df.sample(
        min(3000, len(df)),
        random_state=42
    )

    text_data = sample_df[selected_column].astype(str)

    vectorizer = CountVectorizer(

        stop_words="english",

        max_features=1000

    )

    dtm = vectorizer.fit_transform(text_data)

    lda = LatentDirichletAllocation(

        n_components=5,

        random_state=42

    )

    lda.fit(dtm)

    feature_names = vectorizer.get_feature_names_out()

    topics = []

    for idx, topic in enumerate(lda.components_):

        top_words = [

            feature_names[i]

            for i in topic.argsort()[-10:][::-1]

        ]

        topics.append({

            "Topic": f"Topic {idx+1}",

            "Keywords": ", ".join(top_words)

        })

    topic_df = pd.DataFrame(topics)

    st.dataframe(

        topic_df,

        use_container_width=True

    )

    st.download_button(

        "⬇ Download Topics",

        topic_df.to_csv(index=False),

        "topics.csv",

        "text/csv"

    )

# ==========================================================
# SUMMARIZER
# ==========================================================

elif page == "📝 Summarizer":

    st.title("📝 AI Text Summarizer")

    st.caption(
        "Generate a short summary of any text."
    )

    user_text = st.text_area(

        "Paste text here",

        height=250

    )

    if st.button("✨ Generate Summary"):

        if user_text.strip()=="":

            st.warning(

                "Please enter some text."

            )

        else:

            try:

                parser = PlaintextParser.from_string(

                    user_text,

                    Tokenizer("english")

                )

                summarizer = LsaSummarizer()

                summary = summarizer(

                    parser.document,

                    3

                )

                summary_text = " ".join(

                    str(sentence)

                    for sentence in summary

                )

                if summary_text=="":

                    summary_text="Summary could not be generated."

                st.success(summary_text)

                st.download_button(

                    "⬇ Download Summary",

                    summary_text,

                    file_name="summary.txt"

                )

            except Exception as e:

                st.error(

                    str(e)

                )

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.caption(
"""
🧠 TextLens AI

AI-Powered Text Analytics Platform

Built using Python • Streamlit • Pandas • Plotly • spaCy • NLTK • Gensim • WordCloud • VADER • Sumy
"""
)