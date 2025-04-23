import streamlit as st
import turicreate as tc
import tempfile
import os

st.title("🧠 Sentiment Classifier")

uploaded_file = st.file_uploader("📤 Upload a CSV file with 'review' and optional 'rating' columns", type=["csv"])

if uploaded_file is not None:
    # Save uploaded file to temp path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_csv_path = tmp_file.name

    try:
        # Load and normalize
        sf = tc.SFrame.read_csv(temp_csv_path)
        sf = sf.rename({col: col.strip().lower() for col in sf.column_names()})
        st.success("✅ CSV loaded successfully!")
        st.write(sf.head())

        if "review" not in sf.column_names():
            st.warning("⚠️ No 'review' column found. Please include one.")
        else:
            if "rating" not in sf.column_names():
                st.warning("⚠️ No 'rating' column found. Cannot generate sentiment labels.")
            else:
                sf['sentiment'] = sf['rating'].apply(lambda x: int(x) >= 4)
                sf['word_count'] = tc.text_analytics.count_words(sf['review'])

                # Train + evaluate
                train_data, test_data = sf.random_split(0.8, seed=1)
                model = tc.logistic_classifier.create(
                    train_data,
                    target='sentiment',
                    features=['word_count'],
                    validation_set=None,
                    verbose=False
                )

                eval_results = model.evaluate(test_data)
                auc = eval_results['roc_curve']['auc']
                st.metric(label="📈 AUC Score", value=f"{auc:.4f}")
                st.success("✅ Model trained!")

                # Predict sentiment
                predictions = model.classify(sf)
                sf['predicted_sentiment_score'] = predictions['probability']

                # Most Positive Reviews
                st.subheader("🌟 Most Positive Reviews")
                for row in sf.sort('predicted_sentiment_score', ascending=False).head(3):
                    st.markdown(f"**Score:** {row['predicted_sentiment_score']:.3f}")
                    st.write(row['review'])

                # Most Negative Reviews
                st.subheader("💔 Most Negative Reviews")
                for row in sf.sort('predicted_sentiment_score').head(3):
                    st.markdown(f"**Score:** {row['predicted_sentiment_score']:.3f}")
                    st.write(row['review'])

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("📂 Please upload a CSV to begin.")
