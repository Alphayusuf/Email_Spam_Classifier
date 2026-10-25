import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(
    page_title="Email Spam Detection System",
    page_icon="✉️",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_PATH = Path(__file__).parent / "emails.csv"
LABEL_NAMES = {0: "Legitimate (Ham)", 1: "Spam"}

st.markdown("""
<style>
:root { --navy: #101936; --purple: #6d28d9; --cyan: #06b6d4; --ink: #172033; }
.block-container {padding-top: 1.4rem; padding-bottom: 2.4rem; max-width: 1450px;}
[data-testid="stAppViewContainer"] {background: linear-gradient(135deg, rgba(109,40,217,.045), rgba(6,182,212,.045) 55%, rgba(16,25,54,.025));}
[data-testid="stHeader"] {background: rgba(255,255,255,0);}
[data-testid="stMetric"] {background: linear-gradient(135deg, #ffffff, #f3f5ff); border: 1px solid #e1e7f5; padding: 16px 18px; border-radius: 16px; box-shadow: 0 5px 18px rgba(16,25,54,.06);}
[data-testid="stMetricLabel"] {color: #53617d; font-weight: 650;}
[data-testid="stMetricValue"] {color: #312e81;}
section[data-testid="stSidebar"] {background: linear-gradient(180deg, #101936 0%, #25205b 58%, #164e63 100%);}
section[data-testid="stSidebar"] * {color: #f8fafc !important;}
section[data-testid="stSidebar"] [data-baseweb="select"] * {color: #172033 !important;}
section[data-testid="stSidebar"] [data-testid="stNumberInput"] input {color: #172033 !important;}
.stButton > button, .stDownloadButton > button {border-radius: 10px; font-weight: 650; border: 0; background: linear-gradient(100deg, #6d28d9, #0891b2); color: white;}
.stButton > button:hover, .stDownloadButton > button:hover {border: 0; color: white; filter: brightness(1.08);}
[data-testid="stTabs"] button {font-weight: 650;}
h1 {color: #171b3b; letter-spacing: -.035em;}
h2, h3 {color: #25205b;}
[data-testid="stExpander"] {border: 1px solid #e1e7f5; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_dataset(uploaded_bytes=None):
    if uploaded_bytes is not None:
        from io import BytesIO
        df = pd.read_csv(BytesIO(uploaded_bytes))
        source = "Uploaded CSV"
    else:
        df = pd.read_csv(DATA_PATH)
        source = "Bundled emails.csv"
    if not {"text", "spam"}.issubset(df.columns):
        raise ValueError("Dataset must contain columns named 'text' and 'spam'.")
    df = df[["text", "spam"]].copy()
    df["text"] = df["text"].fillna("").astype(str)
    df["spam"] = pd.to_numeric(df["spam"], errors="coerce")
    df = df.dropna(subset=["spam"])
    df = df[df["spam"].isin([0, 1])]
    df["spam"] = df["spam"].astype(int)
    df = df[df["text"].str.strip().ne("")]
    if df.empty or df["spam"].nunique() != 2:
        raise ValueError("After cleaning, the dataset must contain non-empty text and both labels 0 and 1.")
    return df.reset_index(drop=True), source

def make_pipeline(model_name, max_features, ngram_choice, min_df, alpha, c_value):
    ngram_range = (1, 1) if ngram_choice == "Unigrams" else (1, 2)
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=None if max_features == 0 else max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=True
    )
    if model_name == "Naive Bayes":
        clf = MultinomialNB(alpha=alpha)
    else:
        clf = LinearSVC(C=c_value, class_weight=None)
    return Pipeline([("tfidf", vectorizer), ("classifier", clf)])

@st.cache_resource(show_spinner="Training both models on the selected settings...")
def fit_experiment(data_json, test_size, random_state, max_features, ngram_choice, min_df, alpha, c_value):
    # Serialize/reconstruct data so Streamlit caching respects dataset changes.
    from io import StringIO
    df = pd.read_json(StringIO(data_json), orient="records")
    X = df["text"].astype(str)
    y = df["spam"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    fitted = {}
    summaries = []
    predictions = {}
    decision_scores = {}
    probabilities = {}
    for name in ["Naive Bayes", "Support Vector Machine (SVM)"]:
        model_key = "Naive Bayes" if name == "Naive Bayes" else "SVM"
        pipe = make_pipeline(model_key, max_features, ngram_choice, min_df, alpha, c_value)
        start = time.perf_counter()
        pipe.fit(X_train, y_train)
        training_seconds = time.perf_counter() - start
        start = time.perf_counter()
        pred = pipe.predict(X_test)
        inference_seconds = time.perf_counter() - start
        fitted[model_key] = pipe
        predictions[model_key] = pred
        metrics = {
            "Model": model_key,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision (spam)": precision_score(y_test, pred, pos_label=1, zero_division=0),
            "Recall (spam)": recall_score(y_test, pred, pos_label=1, zero_division=0),
            "F1-score (spam)": f1_score(y_test, pred, pos_label=1, zero_division=0),
            "Training time (s)": training_seconds,
            "Test prediction time (s)": inference_seconds
        }
        summaries.append(metrics)
        clf = pipe.named_steps["classifier"]
        if hasattr(clf, "predict_proba"):
            probabilities[model_key] = pipe.predict_proba(X_test)[:, list(clf.classes_).index(1)]
        elif hasattr(clf, "decision_function"):
            decision_scores[model_key] = pipe.decision_function(X_test)
    test_df = pd.DataFrame({"text": X_test.reset_index(drop=True), "actual": y_test.reset_index(drop=True)})
    test_df["actual_label"] = test_df["actual"].map(LABEL_NAMES)
    for key, pred in predictions.items():
        test_df[f"{key}_prediction"] = pred
        test_df[f"{key}_predicted_label"] = pd.Series(pred).map(LABEL_NAMES).values
        test_df[f"{key}_correct"] = (test_df["actual"].values == pred)
    return fitted, pd.DataFrame(summaries), test_df, y_test.reset_index(drop=True), predictions, probabilities, decision_scores

def metric_pct(v):
    return f"{v:.2%}"

def draw_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticks([0, 1], ["Legitimate", "Spam"], rotation=15)
    ax.set_yticks([0, 1], ["Legitimate", "Spam"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig, cm

def draw_roc(y_true, score_map):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, scores in score_map.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend()
    fig.tight_layout()
    return fig

def draw_pr(y_true, score_map):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, scores in score_map.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–recall curve")
    ax.legend()
    fig.tight_layout()
    return fig

st.markdown("""
<div style="padding: 1.55rem 1.8rem; border-radius: 22px; margin: .2rem 0 1.2rem 0;
background: linear-gradient(115deg, #101936 0%, #43308c 52%, #087e8b 100%);
box-shadow: 0 12px 30px rgba(25, 30, 80, .18);">
<div style="font-size: .82rem; text-transform: uppercase; letter-spacing: .18em; color: #a5f3fc; font-weight: 750;">SMARTMAIL · INTELLIGENT EMAIL SECURITY</div>
<div style="font-size: 2.2rem; line-height: 1.2; font-weight: 800; color: white; margin-top: .45rem;">Email Spam Detection System</div>
<div style="font-size: 1rem; color: #e0e7ff; margin-top: .55rem;">Compare Naïve Bayes and Support Vector Machine predictions in one place.</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Experiment settings")
    uploaded_dataset = st.file_uploader("Optional: upload a labelled dataset CSV", type=["csv"], help="Required columns: text and spam; spam must be 0 for legitimate and 1 for spam.")
    test_percent = st.slider("Test-set percentage", 10, 40, 20, 5)
    random_state = st.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1)
    st.subheader("TF-IDF settings")
    max_features_ui = st.selectbox("Maximum vocabulary features", ["No limit", "5,000", "10,000", "20,000"], index=2)
    max_features = 0 if max_features_ui == "No limit" else int(max_features_ui.replace(",", ""))
    ngram_choice = st.selectbox("Text features", ["Unigrams", "Unigrams + bigrams"], index=1)
    min_df = st.selectbox("Minimum document frequency", [1, 2, 3], index=0)
    st.subheader("Model settings")
    alpha = st.select_slider("Naïve Bayes smoothing (alpha)", options=[0.1, 0.25, 0.5, 1.0, 2.0], value=1.0)
    c_value = st.select_slider("SVM regularisation (C)", options=[0.1, 0.5, 1.0, 2.0, 5.0], value=1.0)
    st.caption("Changing a setting triggers a new experiment. Keep the final test set separate from parameter tuning.")

try:
    dataset, dataset_source = load_dataset(uploaded_dataset.getvalue() if uploaded_dataset else None)
except Exception as e:
    st.error(f"Could not load the dataset: {e}")
    st.stop()

# Avoid stale cache from a changed CSV by including complete current dataset in cache key.
data_json = dataset.to_json(orient="records")
try:
    models, metrics_df, test_df, y_test, predictions, probabilities, decision_scores = fit_experiment(
        data_json, test_percent / 100, int(random_state), max_features, ngram_choice,
        int(min_df), float(alpha), float(c_value)
    )
except Exception as e:
    st.error(f"The experiment could not run: {e}")
    st.stop()

page = st.sidebar.radio(
    "Navigate",
    ["Research Dashboard", "Dataset Explorer", "Model Laboratory", "Evaluation Centre", "Error Analysis", "Classify Email", "Experiment Reports"]
)

if page == "Research Dashboard":
    st.header("Research Dashboard")
    total = len(dataset)
    spam_count = int((dataset["spam"] == 1).sum())
    ham_count = int((dataset["spam"] == 0).sum())
    train_count = total - len(y_test)
    test_count = len(y_test)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Usable emails", f"{total:,}")
    c2.metric("Spam emails", f"{spam_count:,}")
    c3.metric("Legitimate emails", f"{ham_count:,}")
    c4.metric("Held-out test emails", f"{test_count:,}")
    st.subheader("Model performance on the held-out test set")
    display = metrics_df.copy()
    for col in ["Accuracy", "Precision (spam)", "Recall (spam)", "F1-score (spam)"]:
        display[col] = display[col].map(metric_pct)
    st.dataframe(display, use_container_width=True, hide_index=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    metric_cols = ["Accuracy", "Precision (spam)", "Recall (spam)", "F1-score (spam)"]
    metrics_df.set_index("Model")[metric_cols].plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model performance comparison")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(loc="lower right")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(f"Dataset source in this app: {dataset_source}. Split: {100-test_percent}% training / {test_percent}% testing; stratified; random seed {random_state}.")

elif page == "Dataset Explorer":
    st.header("Dataset Explorer")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(dataset):,}")
    c2.metric("Empty messages removed", f"{int(dataset['text'].str.strip().eq('').sum()):,}")
    c3.metric("Duplicate text rows", f"{int(dataset['text'].duplicated().sum()):,}")
    c4.metric("Unique messages", f"{dataset['text'].nunique():,}")
    st.subheader("Class distribution")
    dist = dataset["spam"].map(LABEL_NAMES).value_counts().rename_axis("Class").reset_index(name="Emails")
    c_left, c_right = st.columns(2)
    with c_left:
        st.dataframe(dist, use_container_width=True, hide_index=True)
    with c_right:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(dist["Class"], dist["Emails"])
        ax.set_ylabel("Number of emails")
        ax.set_title("Label distribution")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    st.subheader("Email length")
    lengths = dataset["text"].str.split().str.len()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(lengths.clip(upper=lengths.quantile(0.99)), bins=35)
    ax.set_xlabel("Word count (capped at 99th percentile for display)")
    ax.set_ylabel("Number of emails")
    ax.set_title("Distribution of email lengths")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.subheader("Preview (first 10 records)")
    preview = dataset.copy()
    preview["class"] = preview["spam"].map(LABEL_NAMES)
    preview["text_preview"] = preview["text"].str.replace(r"\s+", " ", regex=True).str.slice(0, 240)
    st.dataframe(preview[["class", "text_preview"]].head(10), use_container_width=True, hide_index=True)
    st.caption("The displayed duplicate count is descriptive. Duplicate removal should be considered carefully and documented before final evaluation.")

elif page == "Model Laboratory":
    st.header("Model Laboratory")
    st.write("Both models are trained on the same stratified training split with the same TF-IDF feature settings.")
    st.markdown(f"- Training records: **{len(dataset) - len(y_test):,}**\n- Test records: **{len(y_test):,}**\n- TF-IDF features: **{ngram_choice}**\n- Maximum features: **{max_features_ui}**\n- Minimum document frequency: **{min_df}**")
    st.markdown(f"- Naïve Bayes alpha: **{alpha}**\n- SVM C: **{c_value}**")
    st.subheader("Learned vocabulary")
    for name, model in models.items():
        with st.expander(f"{name}: feature information"):
            vocab = model.named_steps["tfidf"].get_feature_names_out()
            st.write(f"Vocabulary size: **{len(vocab):,}**")
            st.write("Sample features:")
            st.write(", ".join(vocab[:50]))
    st.caption("Tip: for a final report, choose settings using the training data and reserve the held-out test set for the final evaluation.")

elif page == "Evaluation Centre":
    st.header("Evaluation Centre")
    st.write("All reported test metrics are calculated from the same held-out test records.")
    selected = st.selectbox("Select model for detailed report", ["Naive Bayes", "SVM"])
    pred = predictions[selected]
    cm_fig, cm = draw_confusion(y_test, pred, f"{selected}: confusion matrix")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.pyplot(cm_fig)
        plt.close(cm_fig)
    with c2:
        st.subheader("Confusion matrix counts")
        st.dataframe(pd.DataFrame(cm, index=["Actual legitimate", "Actual spam"], columns=["Predicted legitimate", "Predicted spam"]), use_container_width=True)
        st.markdown(f"""
        - **True negatives:** {cm[0,0]}
        - **False positives:** {cm[0,1]} — legitimate emails flagged as spam
        - **False negatives:** {cm[1,0]} — spam emails missed
        - **True positives:** {cm[1,1]}
        """)
    st.subheader("Classification report")
    report = classification_report(y_test, pred, labels=[0,1], target_names=["Legitimate", "Spam"], output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)
    score_map = {}
    for key in ["Naive Bayes", "SVM"]:
        if key in probabilities:
            score_map[key] = probabilities[key]
        elif key in decision_scores:
            score_map[key] = decision_scores[key]
    if score_map:
        st.subheader("ROC curve")
        roc_fig = draw_roc(y_test, score_map)
        st.pyplot(roc_fig)
        plt.close(roc_fig)
        st.subheader("Precision–recall curve")
        pr_fig = draw_pr(y_test, score_map)
        st.pyplot(pr_fig)
        plt.close(pr_fig)
    else:
        st.info("Score-based curves are not available for the current model configuration.")

elif page == "Error Analysis":
    st.header("Error Analysis")
    chosen = st.selectbox("Model to inspect", ["Naive Bayes", "SVM"])
    pred_col = f"{chosen}_prediction"
    correct_col = f"{chosen}_correct"
    errors = test_df.loc[~test_df[correct_col]].copy()
    c1, c2, c3 = st.columns(3)
    c1.metric("Test messages", len(test_df))
    c2.metric("Misclassified", len(errors))
    c3.metric("Error rate", f"{len(errors)/max(len(test_df),1):.2%}")
    st.write("Inspect examples to understand model errors. Email text may contain sensitive material; do not upload or publish confidential messages.")
    if errors.empty:
        st.success("No errors were observed in this test split for this model.")
    else:
        errors["Actual class"] = errors["actual"].map(LABEL_NAMES)
        errors["Predicted class"] = errors[pred_col].map(LABEL_NAMES)
        errors["Error type"] = errors.apply(
            lambda r: "False positive" if r["actual"] == 0 and r[pred_col] == 1 else "False negative",
            axis=1
        )
        errors["Text preview"] = errors["text"].str.replace(r"\s+", " ", regex=True).str.slice(0, 500)
        st.dataframe(errors[["Actual class", "Predicted class", "Error type", "Text preview"]], use_container_width=True, hide_index=True)
        st.download_button("Download misclassified examples (CSV)", errors[["actual", pred_col, "Error type", "text"]].to_csv(index=False).encode("utf-8"), file_name=f"{chosen.lower().replace(' ','_')}_errors.csv", mime="text/csv")
    st.subheader("Where do both models disagree?")
    disagree = test_df[test_df["Naive Bayes_prediction"] != test_df["SVM_prediction"]].copy()
    st.metric("Model disagreements", len(disagree))
    if len(disagree):
        disagree["Naïve Bayes"] = disagree["Naive Bayes_prediction"].map(LABEL_NAMES)
        disagree["SVM"] = disagree["SVM_prediction"].map(LABEL_NAMES)
        disagree["Actual class"] = disagree["actual"].map(LABEL_NAMES)
        disagree["Text preview"] = disagree["text"].str.replace(r"\s+", " ", regex=True).str.slice(0, 400)
        st.dataframe(disagree[["Actual class", "Naïve Bayes", "SVM", "Text preview"]], use_container_width=True, hide_index=True)

elif page == "Classify Email":
    st.header("Classify Email")
    st.write("Enter text to see how each trained model classifies it. These are predictions, not a guarantee that a message is safe.")
    email_text = st.text_area("Email subject and body", height=220, placeholder="Paste a sample email here...")
    if st.button("Classify with both models", type="primary"):
        if not email_text.strip():
            st.warning("Enter an email first.")
        else:
            result_cols = st.columns(2)
            predictions_new = {}
            for i, (key, pipe) in enumerate(models.items()):
                pred = int(pipe.predict([email_text])[0])
                predictions_new[key] = pred
                with result_cols[i]:
                    st.subheader(key)
                    if pred == 1:
                        st.error("Prediction: SPAM")
                    else:
                        st.success("Prediction: LEGITIMATE (HAM)")
                    clf = pipe.named_steps["classifier"]
                    if key == "Naive Bayes" and hasattr(clf, "predict_proba"):
                        probability = float(pipe.predict_proba([email_text])[0, list(clf.classes_).index(pred)])
                        st.caption(f"Model probability for its predicted class: {probability:.1%}")
                    elif key == "SVM" and hasattr(clf, "decision_function"):
                        score = float(pipe.decision_function([email_text])[0])
                        st.caption(f"SVM decision score: {score:.3f} (not a probability)")
            if len(set(predictions_new.values())) > 1:
                st.warning("The models disagree. Consider this an example for review, not proof that either prediction is correct.")
            else:
                st.info("Both models agree on this message. Agreement does not guarantee correctness.")

elif page == "Experiment Reports":
    st.header("Experiment Reports")
    st.write("Download the outputs generated by the current run.")
    metrics_csv = metrics_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download model metrics (CSV)", metrics_csv, "model_metrics.csv", "text/csv")
    report_rows = []
    for key in ["Naive Bayes", "SVM"]:
        rep = classification_report(y_test, predictions[key], labels=[0,1], target_names=["Legitimate", "Spam"], output_dict=True, zero_division=0)
        for label, values in rep.items():
            if isinstance(values, dict):
                report_rows.append({"Model": key, "Class/measure": label, **values})
    st.download_button("Download classification reports (CSV)", pd.DataFrame(report_rows).to_csv(index=False).encode("utf-8"), "classification_reports.csv", "text/csv")
    combined = test_df.copy()
    st.download_button("Download held-out predictions (CSV)", combined.to_csv(index=False).encode("utf-8"), "held_out_predictions.csv", "text/csv")
    st.subheader("Experiment configuration")
    config = pd.DataFrame([
        {"Setting": "Dataset source", "Value": dataset_source},
        {"Setting": "Rows after basic cleaning", "Value": len(dataset)},
        {"Setting": "Training proportion", "Value": f"{100-test_percent}%"},
        {"Setting": "Test proportion", "Value": f"{test_percent}%"},
        {"Setting": "Random seed", "Value": int(random_state)},
        {"Setting": "TF-IDF n-gram setting", "Value": ngram_choice},
        {"Setting": "Maximum features", "Value": max_features_ui},
        {"Setting": "Minimum document frequency", "Value": int(min_df)},
        {"Setting": "Naïve Bayes alpha", "Value": float(alpha)},
        {"Setting": "SVM C", "Value": float(c_value)},
    ])
    st.dataframe(config, use_container_width=True, hide_index=True)
    st.download_button("Download experiment settings (CSV)", config.to_csv(index=False).encode("utf-8"), "experiment_settings.csv", "text/csv")

st.divider()
st.caption("SmartMail · Email Spam Detection System")
