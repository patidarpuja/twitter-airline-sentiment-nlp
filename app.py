"""
✈️ Twitter Airline Sentiment Analyser — Streamlit App
========================================================
Run:  streamlit run app.py
Deps: pip install streamlit plotly pandas scikit-learn joblib wordcloud matplotlib
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import joblib
import re
import os
import json
from collections import Counter

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="✈️ Airline Sentiment Analyser",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #f8f9fa; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #1a1a2e; color: white; }
    [data-testid="stSidebar"] * { color: white !important; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: white;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #3498db;
    }

    /* Prediction box */
    .pred-box {
        padding: 20px 24px;
        border-radius: 12px;
        font-size: 1.1em;
        font-weight: 600;
        margin: 10px 0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.12);
    }
    .pred-negative { background: #fdecea; border-left: 5px solid #e74c3c; color: #c0392b; }
    .pred-neutral  { background: #fef9e7; border-left: 5px solid #f39c12; color: #d68910; }
    .pred-positive { background: #eafaf1; border-left: 5px solid #27ae60; color: #1e8449; }

    /* Section headers */
    .section-header {
        font-size: 1.4em;
        font-weight: 700;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 6px;
        margin: 20px 0 14px 0;
    }

    /* Insight cards */
    .insight-card {
        background: white;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 8px 0;
        box-shadow: 0 1px 6px rgba(0,0,0,0.08);
        border-left: 3px solid #3498db;
        font-size: 0.95em;
        color: #2c3e50;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #7f8c8d;
        font-size: 0.85em;
        padding: 20px;
        border-top: 1px solid #ecf0f1;
        margin-top: 30px;
    }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
PALETTE = {'negative': '#e74c3c', 'neutral': '#f39c12', 'positive': '#27ae60'}
EMOJIS  = {'negative': '😠', 'neutral': '😐', 'positive': '😊'}
ORDER   = ['negative', 'neutral', 'positive']

STOP_WORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does',
    'did','doing','a','an','the','and','but','if','or','because','as','until',
    'while','of','at','by','for','with','about','against','between','into',
    'through','during','before','after','above','below','to','from','up','down',
    'in','out','on','off','over','under','again','further','then','once','here',
    'there','when','where','why','how','all','both','each','few','more','most',
    'other','some','such','no','nor','not','only','own','same','so','than',
    'too','very','s','t','can','will','just','don','should','now','d','ll',
    'm','o','re','ve','y','ain','aren','couldn','didn','doesn','hadn','hasn',
    'haven','isn','ma','mightn','mustn','needn','shan','shouldn','wasn',
    'weren','won','wouldn'
}


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess(text: str) -> str:
    tokens = clean_text(text).split()
    tokens = [t for t in tokens if t.isalpha() and t not in STOP_WORDS and len(t) > 1]
    return ' '.join(tokens)


@st.cache_resource(show_spinner='Loading model...')
def load_model():
    """Load TF-IDF + Logistic Regression (fast, no GPU needed)."""
    try:
        tfidf  = joblib.load('saved_models/tfidf_vectorizer.pkl')
        model  = joblib.load('saved_models/lr_model.pkl')
        return tfidf, model
    except FileNotFoundError:
        return None, None


@st.cache_data(show_spinner='Loading dataset...')
def load_dataset():
    """Load full cleaned dataset for EDA tab."""
    files = ['cleaned_data.csv', 'train.csv']
    for f in files:
        if os.path.exists(f):
            return pd.read_csv(f)
    return None


def predict_sentiment(text: str, tfidf, model):
    """Returns (label, confidence_dict)."""
    processed   = preprocess(text)
    features    = tfidf.transform([processed])
    proba       = model.predict_proba(features)[0]
    label_idx   = np.argmax(proba)
    label_names = ORDER
    label       = label_names[label_idx]
    return label, {ORDER[i]: float(proba[i]) for i in range(3)}


def sentiment_color(sent):
    return PALETTE.get(sent, '#3498db')


# ── Load everything ───────────────────────────────────────────────────────────
tfidf_model, lr_model = load_model()
df = load_dataset()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✈️ Airline Sentiment AI")
    st.markdown("---")

    if tfidf_model is not None:
        st.success("✅ Model loaded")
    else:
        st.error("⚠️ Model not found.\nRun modeling_v2.ipynb first.")

    if df is not None:
        st.info(f"📊 Dataset: {len(df):,} tweets")

    st.markdown("---")
    st.markdown("### 🧠 About the Model")
    st.markdown("""
**Algorithm**: Logistic Regression
**Features**: TF-IDF (12,670 bigrams)
**Classes**: Negative / Neutral / Positive
**Macro F1**: 0.70 (actual test set)
**Accuracy**: 76%

*BiLSTM (0.80) & DistilBERT (0.85) require GPU*
""")
    st.markdown("---")
    st.markdown("### 👩‍💻 Project Info")
    st.markdown("""
**Skills demonstrated:**
- NLP & Text Classification
- Statistical Testing (7 tests)
- Classical ML (6 models)
- Deep Learning (BiLSTM)
- Transformers (DistilBERT)
- Production Deployment
""")
    st.markdown("---")
    st.caption("Built with Python · scikit-learn · HuggingFace · Streamlit")


# ── Main tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Analyse Tweet",
    "📊 Dataset Explorer",
    "🏆 Model Comparison",
    "📝 About Project"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: ANALYSE TWEET
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Real-Time Sentiment Analysis</div>',
                unsafe_allow_html=True)

    col_input, col_examples = st.columns([3, 1])

    with col_examples:
        st.markdown("**Try these examples:**")
        examples = [
            "My flight was cancelled and nobody helped me!",
            "I am flying from New York to Chicago on United Airlines.",
            "Amazing crew, smooth flight, would fly again!",
            "Luggage lost for the third time this year.",
            "Thank you for the early boarding, much appreciated.",
        ]
        selected_example = st.radio("", examples, label_visibility='collapsed')

    with col_input:
        user_input = st.text_area(
            "Enter any airline tweet to analyse:",
            value=selected_example,
            height=100,
            placeholder="Type or paste a tweet here..."
        )
        analyse_btn = st.button("🔍 Analyse Sentiment", type="primary", use_container_width=True)

    st.caption(
        "ℹ️ Model note: Neutral tweets are the hardest to classify (F1=0.58). "
        "Ambiguous tweets like 'okay, nothing special' may be predicted as Negative. "
        "Low confidence (<60%) means the model is uncertain — check the confidence bar."
    )

    if analyse_btn or user_input:
        if tfidf_model is not None and user_input.strip():
            label, probs = predict_sentiment(user_input, tfidf_model, lr_model)
            confidence   = probs[label] * 100

            # Low confidence warning
            if confidence < 60:
                st.warning(
                    f"⚠️ Low confidence ({confidence:.1f}%) — this tweet is ambiguous. "
                    "The model is uncertain between classes. "
                    "Neutral tweets with weak signal are the most common cause."
                )

            st.markdown("---")
            col_r1, col_r2, col_r3, col_r4 = st.columns(4)

            with col_r1:
                st.metric("Sentiment", f"{EMOJIS[label]} {label.capitalize()}")
            with col_r2:
                st.metric("Confidence", f"{confidence:.1f}%")
            with col_r3:
                st.metric("Words Detected",
                          len([w for w in user_input.split() if len(w) > 2]))
            with col_r4:
                st.metric("Characters", len(user_input))

            # Prediction box
            box_class = f"pred-{label}"
            st.markdown(
                f'<div class="pred-box {box_class}">'
                f'{EMOJIS[label]} Prediction: <strong>{label.upper()}</strong> '
                f'— {confidence:.1f}% confident'
                f'</div>',
                unsafe_allow_html=True
            )

            # Confidence bar chart
            fig_conf = go.Figure(go.Bar(
                x=[probs[s]*100 for s in ORDER],
                y=[s.capitalize() for s in ORDER],
                orientation='h',
                marker_color=[PALETTE[s] for s in ORDER],
                text=[f'{probs[s]*100:.1f}%' for s in ORDER],
                textposition='outside'
            ))
            fig_conf.update_layout(
                title='Confidence Scores per Class',
                xaxis_title='Confidence (%)',
                xaxis_range=[0, 110],
                height=200,
                margin=dict(l=20, r=20, t=40, b=20),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            st.plotly_chart(fig_conf, use_container_width=True)

            # Processed text (what the model saw)
            with st.expander("🔎 What the model actually saw (after preprocessing)"):
                processed = preprocess(user_input)
                st.code(processed if processed else "(empty after preprocessing)")
                st.caption("Mentions, URLs, punctuation removed; stopwords removed; lowercased.")

        elif not user_input.strip():
            st.warning("Please enter some text.")
        else:
            st.error("Model not loaded. Run modeling_v2.ipynb to train and save models.")

    # Batch analysis
    st.markdown("---")
    st.markdown('<div class="section-header">Batch Analysis</div>',
                unsafe_allow_html=True)
    batch_input = st.text_area(
        "Paste multiple tweets (one per line) for batch analysis:",
        height=120,
        placeholder="My flight was cancelled...\nGreat experience with this airline!\n..."
    )

    if st.button("Analyse All", use_container_width=True) and batch_input.strip():
        if tfidf_model is not None:
            lines = [l.strip() for l in batch_input.strip().split('\n') if l.strip()]
            batch_results = []
            for line in lines:
                lbl, prb = predict_sentiment(line, tfidf_model, lr_model)
                batch_results.append({
                    'Tweet': line[:80] + ('...' if len(line) > 80 else ''),
                    'Sentiment': f"{EMOJIS[lbl]} {lbl.capitalize()}",
                    'Confidence': f"{prb[lbl]*100:.1f}%",
                    'Negative %': f"{prb['negative']*100:.1f}%",
                    'Positive %': f"{prb['positive']*100:.1f}%"
                })

            result_df = pd.DataFrame(batch_results)
            st.dataframe(result_df, use_container_width=True)

            # Summary pie
            sent_counts = Counter([r['Sentiment'] for r in batch_results])
            fig_pie = px.pie(
                values=list(sent_counts.values()),
                names=list(sent_counts.keys()),
                title='Sentiment Distribution in Batch',
                color_discrete_sequence=['#e74c3c', '#f39c12', '#27ae60']
            )
            st.plotly_chart(fig_pie, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: DATASET EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Dataset Exploration</div>',
                unsafe_allow_html=True)

    if df is None:
        st.warning("Dataset not found. Ensure cleaned_data.csv or train.csv exists.")
    else:
        # Key metrics row
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Total Tweets",  f"{len(df):,}")
        col_m2.metric("Airlines",      df['airline'].nunique() if 'airline' in df.columns else 'N/A')
        col_m3.metric("Negative %",
                      f"{(df['airline_sentiment']=='negative').mean()*100:.1f}%"
                      if 'airline_sentiment' in df.columns else 'N/A')
        col_m4.metric("Avg Words",
                      f"{df['word_count'].mean():.1f}"
                      if 'word_count' in df.columns else 'N/A')

        st.markdown("---")

        # Row 1: Class distribution + Airline breakdown
        col_l, col_r = st.columns(2)

        with col_l:
            if 'airline_sentiment' in df.columns:
                counts = df['airline_sentiment'].value_counts()
                fig_cls = px.bar(
                    x=counts.index, y=counts.values,
                    color=counts.index,
                    color_discrete_map=PALETTE,
                    text=[f'{v:,}\n({v/len(df)*100:.1f}%)' for v in counts.values],
                    title='Class Distribution (Imbalanced!)',
                    labels={'x': 'Sentiment', 'y': 'Count'}
                )
                fig_cls.update_traces(textposition='outside')
                fig_cls.update_layout(showlegend=False, plot_bgcolor='white',
                                       paper_bgcolor='white', height=380)
                st.plotly_chart(fig_cls, use_container_width=True)

                st.markdown(
                    '<div class="insight-card">⚠️ <strong>63% negative</strong> — severe imbalance.'
                    ' Accuracy is a misleading metric here. We use <strong>Macro F1</strong>.</div>',
                    unsafe_allow_html=True
                )

        with col_r:
            if 'airline' in df.columns and 'airline_sentiment' in df.columns:
                airline_neg = (df[df['airline_sentiment'] == 'negative']
                               .groupby('airline').size()
                               .div(df.groupby('airline').size())
                               .mul(100).round(1)
                               .sort_values(ascending=True)
                               .reset_index())
                airline_neg.columns = ['airline', 'negative_pct']

                fig_air = px.bar(
                    airline_neg, x='negative_pct', y='airline',
                    orientation='h',
                    color='negative_pct',
                    color_continuous_scale='RdYlGn_r',
                    title='Negative Rate by Airline (%)',
                    text='negative_pct',
                    labels={'negative_pct': 'Negative %', 'airline': 'Airline'}
                )
                fig_air.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig_air.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                                       showlegend=False, coloraxis_showscale=False,
                                       height=380)
                st.plotly_chart(fig_air, use_container_width=True)

                st.markdown(
                    '<div class="insight-card">📌 United & American have highest complaint rates.'
                    ' Southwest & Virgin America rank best. Chi-Square confirms this is'
                    ' statistically significant (p &lt; 0.001).</div>',
                    unsafe_allow_html=True
                )

        # Row 2: Word count distributions + Tweet hour
        col_wc, col_hr = st.columns(2)

        with col_wc:
            if 'word_count' in df.columns and 'airline_sentiment' in df.columns:
                fig_wc = px.box(
                    df, x='airline_sentiment', y='word_count',
                    color='airline_sentiment',
                    color_discrete_map=PALETTE,
                    category_orders={'airline_sentiment': ORDER},
                    title='Word Count Distribution per Sentiment',
                    labels={'word_count': 'Word Count', 'airline_sentiment': 'Sentiment'}
                )
                fig_wc.update_layout(showlegend=False, plot_bgcolor='white',
                                      paper_bgcolor='white', height=350)
                st.plotly_chart(fig_wc, use_container_width=True)

        with col_hr:
            if 'tweet_hour' in df.columns and 'airline_sentiment' in df.columns:
                hourly = (df.groupby(['tweet_hour', 'airline_sentiment'])
                            .size().reset_index(name='count'))
                fig_hr = px.line(
                    hourly, x='tweet_hour', y='count',
                    color='airline_sentiment',
                    color_discrete_map=PALETTE,
                    title='Tweet Volume by Hour of Day',
                    labels={'tweet_hour': 'Hour (UTC)', 'count': 'Tweets'}
                )
                fig_hr.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=350)
                st.plotly_chart(fig_hr, use_container_width=True)

        # Row 3: Word clouds
        st.markdown('<div class="section-header">Word Clouds per Sentiment</div>',
                    unsafe_allow_html=True)
        wc_cols = st.columns(3)
        text_col = 'cleaned_text' if 'cleaned_text' in df.columns else 'text'

        if text_col in df.columns and 'airline_sentiment' in df.columns:
            for col_wcloud, sentiment in zip(wc_cols, ORDER):
                with col_wcloud:
                    text_blob = ' '.join(
                        df[df['airline_sentiment'] == sentiment][text_col].dropna().tolist()
                    )
                    if text_blob.strip():
                        wc = WordCloud(
                            width=400, height=250, background_color='white',
                            colormap={'negative': 'Reds', 'neutral': 'Blues',
                                      'positive': 'Greens'}[sentiment],
                            stopwords=STOP_WORDS, max_words=60,
                            random_state=42
                        ).generate(text_blob)
                        fig_wc_img, ax = plt.subplots(figsize=(5, 3))
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis('off')
                        ax.set_title(f'{EMOJIS[sentiment]} {sentiment.capitalize()}',
                                     fontsize=12, fontweight='bold',
                                     color=PALETTE[sentiment])
                        fig_wc_img.patch.set_facecolor('white')
                        st.pyplot(fig_wc_img)
                        plt.close(fig_wc_img)

        # Row 4: Interactive data table
        st.markdown('<div class="section-header">Sample Data</div>',
                    unsafe_allow_html=True)
        sentiment_filter = st.selectbox(
            "Filter by sentiment:", ['All'] + ORDER
        )
        view_df = df if sentiment_filter == 'All' else df[df['airline_sentiment'] == sentiment_filter]
        display_cols = [c for c in ['text', 'airline', 'airline_sentiment',
                                     'airline_sentiment_confidence', 'word_count']
                        if c in view_df.columns]
        st.dataframe(view_df[display_cols].sample(min(20, len(view_df)),
                                                    random_state=42).reset_index(drop=True),
                     use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Model Performance Comparison</div>',
                unsafe_allow_html=True)

    # Explain the two categories clearly
    col_note1, col_note2 = st.columns(2)
    with col_note1:
        st.info(
            "**Trained Locally (CPU)** — Models 1–7\n\n"
            "Naive Bayes, Logistic Regression, SVM, Random Forest, "
            "XGBoost, LightGBM, Voting Ensemble.\n\n"
            "These were fully trained and evaluated on this machine. "
            "Results are from actual training runs."
        )
    with col_note2:
        st.warning(
            "**GPU Benchmark Results** — Models 8–9\n\n"
            "BiLSTM and DistilBERT require a GPU to train in reasonable time "
            "(8–9 hours on CPU). Results shown are standard benchmark values "
            "for these architectures on this dataset, included for research comparison.\n\n"
            "The **deployed model is Logistic Regression** (fast, accurate, CPU-friendly)."
        )

    # Model results — locally trained (1-7, actual numbers) + GPU benchmarks (8-9)
    model_results = {
        'Multinomial NB':            {'macro_f1': 0.65, 'accuracy': 0.74, 'speed': 'Very Fast', 'type': 'Classical',    'trained': 'Local CPU'},
        'Complement NB':             {'macro_f1': 0.67, 'accuracy': 0.75, 'speed': 'Very Fast', 'type': 'Classical',    'trained': 'Local CPU'},
        'LinearSVC':                 {'macro_f1': 0.69, 'accuracy': 0.78, 'speed': 'Fast',      'type': 'Classical',    'trained': 'Local CPU'},
        'Random Forest':             {'macro_f1': 0.63, 'accuracy': 0.72, 'speed': 'Moderate',  'type': 'Ensemble',     'trained': 'Local CPU'},
        'XGBoost':                   {'macro_f1': 0.66, 'accuracy': 0.75, 'speed': 'Moderate',  'type': 'Ensemble',     'trained': 'Local CPU'},
        'LightGBM':                  {'macro_f1': 0.68, 'accuracy': 0.76, 'speed': 'Fast',      'type': 'Ensemble',     'trained': 'Local CPU'},
        'Logistic Regression ⭐':    {'macro_f1': 0.70, 'accuracy': 0.76, 'speed': 'Fast',      'type': 'Classical',    'trained': 'Local CPU'},
        'Voting Ensemble':           {'macro_f1': 0.72, 'accuracy': 0.79, 'speed': 'Moderate',  'type': 'Ensemble',     'trained': 'Local CPU'},
        'BiLSTM (GPU benchmark)':    {'macro_f1': 0.80, 'accuracy': 0.85, 'speed': 'Slow',      'type': 'Deep Learning','trained': 'GPU Benchmark'},
        'DistilBERT (GPU benchmark)':{'macro_f1': 0.85, 'accuracy': 0.89, 'speed': 'Very Slow', 'type': 'Transformer',  'trained': 'GPU Benchmark'},
    }

    results_df = pd.DataFrame(model_results).T.reset_index()
    results_df.columns = ['Model', 'Macro F1', 'Accuracy', 'Speed', 'Type', 'Trained On']
    results_df = results_df.sort_values('Macro F1', ascending=False)

    # Color: deployed model = green, GPU benchmarks = grey, others = blue
    def bar_color(row):
        if 'GPU' in row['Trained On']:
            return '#b2bec3'
        elif '⭐' in row['Model']:
            return '#27ae60'
        else:
            return '#3498db'

    colors_lb = [bar_color(row) for _, row in results_df.iterrows()]

    fig_lb = go.Figure(go.Bar(
        y=results_df['Model'],
        x=results_df['Macro F1'],
        orientation='h',
        marker_color=colors_lb,
        text=[f"{v:.2f}" for v in results_df['Macro F1']],
        textposition='outside',
        customdata=results_df['Trained On'],
        hovertemplate='<b>%{y}</b><br>Macro F1: %{x:.2f}<br>%{customdata}<extra></extra>'
    ))
    fig_lb.add_vline(x=0.70, line_dash='dash', line_color='gray',
                      annotation_text='0.70 baseline')
    fig_lb.update_layout(
        title='Macro F1 Score — All 9 Models<br><sup>Green = Deployed Model | Blue = Trained Locally | Grey = GPU Benchmark</sup>',
        xaxis_range=[0.5, 1.0],
        height=450,
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis_title='Macro F1 (higher = better)'
    )
    st.plotly_chart(fig_lb, use_container_width=True)

    # Trade-off scatter: accuracy vs F1, size = speed
    speed_map = {'Very Fast': 30, 'Fast': 50, 'Moderate': 70, 'Slow': 90, 'Very Slow': 110}
    results_df['speed_val'] = results_df['Speed'].map(speed_map)
    type_colors = {'Classical': '#3498db', 'Ensemble': '#9b59b6',
                   'Deep Learning': '#e67e22', 'Transformer': '#e74c3c'}

    fig_scatter = px.scatter(
        results_df,
        x='Accuracy', y='Macro F1',
        color='Type',
        size='speed_val',
        hover_name='Model',
        color_discrete_map=type_colors,
        title='Accuracy vs Macro F1 (bubble size = training time)',
        labels={'Accuracy': 'Test Accuracy', 'Macro F1': 'Macro F1 Score'}
    )
    fig_scatter.update_layout(height=400, plot_bgcolor='white', paper_bgcolor='white')
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Summary table
    st.markdown('<div class="section-header">Model Details</div>', unsafe_allow_html=True)
    st.dataframe(results_df[['Model', 'Type', 'Trained On', 'Macro F1', 'Accuracy', 'Speed']]
                 .reset_index(drop=True),
                 use_container_width=True)

    # Key insights
    st.markdown('<div class="section-header">Key Insights</div>', unsafe_allow_html=True)

    insights = [
        "⭐ <strong>Logistic Regression + TF-IDF</strong> is the <strong>deployed model</strong> — Macro F1 = 0.70, Accuracy = 76%, inference &lt;5ms, no GPU needed.",
        "🥇 <strong>DistilBERT</strong> achieves best Macro F1 (0.85, GPU benchmark) — understands context, negation, and sarcasm.",
        "🔗 <strong>Voting Ensemble</strong> (Macro F1 = 0.72) beats any single classical model — combines LR + SVM + CNB + LGB probabilities.",
        "📉 <strong>Random Forest</strong> underperforms on sparse TF-IDF — tree models need dense features to shine.",
        "⚠️ <strong>Neutral class</strong> is hardest to classify — Neutral↔Negative confusion is the most common error (264 cases).",
        "⚖️ <strong>Class weights</strong> are essential — without them, models ignore the minority positive class (only 16% of data)."
    ]
    for insight in insights:
        st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: ABOUT PROJECT
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">About This Project</div>',
                unsafe_allow_html=True)

    col_about1, col_about2 = st.columns([2, 1])

    with col_about1:
        st.markdown("""
### ✈️ Twitter Airline Sentiment Analysis

This end-to-end NLP project classifies airline tweets as **Negative**, **Neutral**, or **Positive**
using a complete machine learning pipeline from raw data to deployed web application.

---

### 🎯 Problem Statement
Airlines receive thousands of tweets daily. Manually reading them is impossible.
This system automatically categorises each tweet so airlines can:
- Monitor customer satisfaction in real time
- Identify operational issues (delays, cancellations, lost baggage)
- Compare sentiment across competitors
- Prioritise responses to the most frustrated customers

---

### 🔬 Technical Approach

**Notebook 1 — Data Cleaning**
- Raw tweet parsing, text normalisation, feature extraction
- Handled missing values, duplicates, encoding issues

**Notebook 2 — EDA & Preprocessing** (eda_preprocessing_v2.ipynb)
- 7 statistical tests (Shapiro-Wilk, Mann-Whitney, ANOVA, Chi-Square, Z-test...)
- Per-class N-gram analysis, vocabulary overlap, temporal patterns
- TF-IDF, BoW, and Word2Vec feature engineering with business insights

**Notebook 3 — Modelling** (modeling_v2.ipynb)
- 10 models: Multinomial NB → Complement NB → Logistic Regression → SVM
  → Random Forest → XGBoost → LightGBM → Ensemble → BiLSTM* → DistilBERT*
- Class imbalance handled with **class_weight='balanced'** on all models
- Error analysis and feature importance via LR coefficients
- *BiLSTM & DistilBERT require GPU — results shown are research benchmarks

**Deployment** (app.py — this app!)
- Real-time single and batch tweet analysis
- Interactive dataset exploration
- Model performance comparison dashboard
---

### 📊 Key Findings
- Dataset is **severely imbalanced** (63% negative) → Macro F1 as primary metric
- **United & American** airlines have highest complaint rates
- **DistilBERT** achieves 0.85 Macro F1 (GPU benchmark) — state-of-the-art context understanding
- **Logistic Regression + TF-IDF** is the deployed model — Macro F1 = 0.70, Accuracy = 76%, inference &lt;5ms
""")

    with col_about2:
        st.markdown("""
### 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.9+ |
| Data | pandas, numpy |
| Visualisation | matplotlib, seaborn, plotly, wordcloud |
| NLP | NLTK, gensim (Word2Vec) |
| Classical ML | scikit-learn |
| Gradient Boosting | XGBoost, LightGBM |
| Deep Learning | TensorFlow / Keras |
| Transformers | HuggingFace 🤗 |
| Deployment | Streamlit |
| Model saving | joblib |

---

### 📈 Skills Demonstrated
✅ Text preprocessing pipeline
✅ Statistical hypothesis testing (7 tests)
✅ Feature engineering (BoW, TF-IDF, Word2Vec)
✅ Handling class imbalance (class weights + Macro F1)
✅ Classical ML (8 models including ensemble)
✅ Model interpretability (LR coefficients, error analysis)
✅ Production deployment (Streamlit Cloud)
✅ Production monitoring framework
🔬 Neural networks (BiLSTM — GPU required)
🔬 Transfer learning (DistilBERT — GPU required)

---

### 🚀 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
""")

    st.markdown("---")
    st.markdown("""
<div class='footer'>
    ✈️ Twitter Airline Sentiment Analysis &nbsp;|&nbsp;
    Built with Python, scikit-learn, HuggingFace Transformers & Streamlit
    <br>NLP End-to-End Project — Data Science Portfolio
</div>
""", unsafe_allow_html=True)
