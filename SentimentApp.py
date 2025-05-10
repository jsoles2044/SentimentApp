import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

st.title("🧠 Sentiment Classifier (scikit-learn)")

uploaded_file = st.file_uploader("📤 Upload a CSV file with 'review' and optional 'rating' columns", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = [col.strip().lower() for col in df.columns]
        st.success("✅ CSV loaded successfully!")
        st.write(df.head())

        if "review" not in df.columns:
            st.warning("⚠️ No 'review' column found. Please include one.")
        else:
            if "rating" not in df.columns:
                st.warning("⚠️ No 'rating' column found. Cannot generate sentiment labels.")
            else:
                df['sentiment'] = df['rating'].apply(lambda x: int(x) >= 4)

                X = df['review']
                y = df['sentiment']

                vectorizer = CountVectorizer()
                X_vectorized = vectorizer.fit_transform(X)

                X_train, X_test, y_train, y_test = train_test_split(
                    X_vectorized, y, test_size=0.2, random_state=1)

                model = LogisticRegression(max_iter=1000)
                model.fit(X_train, y_train)

                y_pred_proba = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_pred_proba)

                st.metric(label="📈 AUC Score", value=f"{auc:.4f}")
                st.success("✅ Model trained!")

                # Predict full dataset
                full_probs = model.predict_proba(X_vectorized)[:, 1]
                df['predicted_sentiment_score'] = full_probs

                # Most Positive Reviews
                st.subheader("🌟 Most Positive Reviews")
                for _, row in df.sort_values('predicted_sentiment_score', ascending=False).head(3).iterrows():
                    st.markdown(f"**Score:** {row['predicted_sentiment_score']:.3f}")
                    st.write(row['review'])

                # Most Negative Reviews
                st.subheader("💔 Most Negative Reviews")
                for _, row in df.sort_values('predicted_sentiment_score').head(3).iterrows():
                    st.markdown(f"**Score:** {row['predicted_sentiment_score']:.3f}")
                    st.write(row['review'])

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("📂 Please upload a CSV to begin.")
