
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Lab 8 | Product Feedback Analysis",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
[data-testid="stMetric"] {
    background: #f3f6fb;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #e5eaf2;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# LESSON 1: PREPARING QUALITATIVE DATA
# ==========================================

reviews = [
    "The bottle keeps my water cold all day.",
    "Great size and easy to carry to class.",
    "I love the simple design and sturdy lid.",
    "It leaks in my bag. Very disappointing.",
    "The cap is hard to open and the bottle feels heavy.",
    "Good value, but cleaning the narrow opening is difficult.",
    "Keeps drinks cold, though the paint scratches easily.",
    "The bottle is fine. Nothing special either way.",
    "I expected better insulation for this price.",
    "Lightweight, useful, and easy to wash.",
    "Not bad, but I would prefer a larger capacity.",
    "Excellent bottle. I use it every day.",
    "The lid broke after one week. Poor quality.",
    "It works as described and arrived on time.",
    "Nice color, but it does not keep drinks cold.",
    "Affordable and practical for carrying water.",
    "The bottle is okay; the handle is uncomfortable.",
    "Very happy with the purchase and the quality."
]

# Main topic assigned to each comment
topics = [
    "Insulation",
    "Design and Portability",
    "Lid and Usability",
    "Lid and Usability",
    "Lid and Usability",
    "Cleaning",
    "Durability",
    "Overall Experience",
    "Insulation",
    "Cleaning",
    "Design and Portability",
    "Overall Experience",
    "Lid and Usability",
    "Delivery",
    "Insulation",
    "Price and Value",
    "Lid and Usability",
    "Overall Experience"
]

df = pd.DataFrame({
    "ID": [f"R{i:02}" for i in range(1, 19)],
    "Comment": reviews,
    "Topic": topics
})

# ==========================================
# SENTIMENT ANALYSIS USING VADER
# ==========================================

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]

    if score >= 0.05:
        return "Positive", score
    elif score <= -0.05:
        return "Negative", score
    else:
        return "Neutral", score

df[["Sentiment", "Compound Score"]] = df["Comment"].apply(
    lambda text: pd.Series(analyze_sentiment(text))
)

sentiment_order = ["Positive", "Neutral", "Negative"]

colors = {
    "Positive": "#159895",
    "Neutral": "#E0A43B",
    "Negative": "#D66B70"
}

# ==========================================
# DASHBOARD HEADER
# ==========================================

st.title("Product Feedback Analysis Dashboard")

st.write(
    "This dashboard analyzes 18 product reviews using "
    "VADER sentiment analysis and qualitative data visualization."
)

st.caption(
    "TFB3133 / TEB3133 Data Visualization | "
    "Lab 8: Qualitative Data and Sentiment Analysis"
)

# ==========================================
# SIDEBAR FILTERS
# ==========================================

st.sidebar.header("Dashboard Filters")

selected_sentiments = st.sidebar.multiselect(
    "Select Sentiment",
    options=sentiment_order,
    default=sentiment_order
)

selected_topics = st.sidebar.multiselect(
    "Select Topic",
    options=sorted(df["Topic"].unique()),
    default=sorted(df["Topic"].unique())
)

search_text = st.sidebar.text_input(
    "Search Comments",
    placeholder="Enter a keyword..."
)

filtered_df = df[
    df["Sentiment"].isin(selected_sentiments) &
    df["Topic"].isin(selected_topics)
].copy()

if search_text.strip():
    filtered_df = filtered_df[
        filtered_df["Comment"].str.contains(
            search_text.strip(),
            case=False,
            regex=False
        )
    ]

# ==========================================
# OVERVIEW METRICS
# ==========================================

counts = (
    filtered_df["Sentiment"]
    .value_counts()
    .reindex(sentiment_order, fill_value=0)
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Reviews", len(filtered_df))
col2.metric("Positive Reviews", int(counts["Positive"]))
col3.metric("Neutral Reviews", int(counts["Neutral"]))
col4.metric("Negative Reviews", int(counts["Negative"]))

st.divider()

# ==========================================
# DASHBOARD TABS
# ==========================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Original Comments",
    "Sentiment Audit",
    "Topic Analysis",
    "Assignment Report"
])

# ==========================================
# TAB 1: SENTIMENT CHART AND WORD CLOUD
# ==========================================

with tab1:

    st.header("Sentiment Overview")

    if filtered_df.empty:
        st.warning(
            "No reviews match the selected filters."
        )

    else:

        left, right = st.columns(2)

        with left:

            st.subheader("Sentiment Distribution")

            fig, ax = plt.subplots(figsize=(8, 5))

            bars = ax.barh(
                sentiment_order,
                counts.values,
                color=[
                    colors[s] for s in sentiment_order
                ]
            )

            ax.invert_yaxis()
            ax.set_xlabel("Number of Reviews")
            ax.set_ylabel("Sentiment")
            ax.set_title("Comment Counts by Sentiment")

            ax.set_xlim(
                0,
                max(2, int(counts.max()) + 2)
            )

            ax.bar_label(bars, padding=4)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "The chart compares the number of comments "
                "classified as positive, neutral, and negative."
            )

        with right:

            st.subheader("Word Cloud")

            exclude_bottle = st.checkbox(
                "Exclude 'bottle' from the word cloud",
                value=False
            )

            # Original custom stop words:
            # bottle, drinks, drink
            #
            # Assignment improvement:
            # Remove 'bottle' from the original stop-word
            # list and compare the visualization.

            stop_words = set(STOPWORDS) | {
                "drinks",
                "drink"
            }

            if exclude_bottle:
                stop_words.add("bottle")

            text_data = " ".join(
                filtered_df["Comment"]
            )

            cloud = WordCloud(
                width=900,
                height=500,
                background_color="white",
                colormap="viridis",
                stopwords=stop_words,
                collocations=False,
                random_state=42
            ).generate(text_data)

            fig, ax = plt.subplots(figsize=(8, 5))

            ax.imshow(
                cloud,
                interpolation="bilinear"
            )

            ax.axis("off")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "Larger words appear more frequently. "
                "Word size does not indicate sentiment "
                "or importance. Color and position do "
                "not represent sentiment."
            )

    st.divider()

    st.subheader("Word Cloud Improvement")

    st.write(
        "The word 'bottle' was removed from the original "
        "custom stop-word list to examine how a frequent "
        "product-related word affects the visualization. "
        "The checkbox allows readers to compare the results."
    )

    st.write(
        "The word 'lid' was retained because it represents "
        "a useful product feature. Its appearance can draw "
        "attention to comments discussing lid durability, "
        "leaking, and usability."
    )

    st.subheader("Mixed-Tone Comment")

    st.info(
        "R06: Good value, but cleaning the narrow "
        "opening is difficult."
    )

    st.write(
        "This comment contains both positive and negative "
        "feedback. The customer appreciates the product's "
        "value but finds cleaning difficult. A single "
        "sentiment label cannot fully represent both opinions."
    )

# ==========================================
# TAB 2: ORIGINAL COMMENTS
# ==========================================

with tab2:

    st.header("Original Customer Feedback")

    st.write(
        "The original comments are retained to allow "
        "manual verification of the automated results."
    )

    st.dataframe(
        filtered_df[
            [
                "ID",
                "Comment",
                "Topic",
                "Sentiment",
                "Compound Score"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "VADER compound scores range from -1 to +1. "
        "Scores are not confidence ratings."
    )

    st.download_button(
        label="Download Analyzed Data (CSV)",
        data=filtered_df.to_csv(index=False),
        file_name="lab8_analyzed_feedback.csv",
        mime="text/csv"
    )

# ==========================================
# TAB 3: MANUAL SENTIMENT AUDIT
# ==========================================

with tab3:

    st.header("Manual Sentiment Audit")

    st.write(
        "Six comments are reviewed: two positive, "
        "two neutral, and two negative based on "
        "their automated sentiment labels."
    )

    # Independent human interpretations
    human_review = {
        "R01": ("Positive", "Good insulation."),
        "R02": ("Positive", "Size and portability are praised."),
        "R03": ("Positive", "Design and lid are praised."),
        "R04": ("Negative", "Leaking is disappointing."),
        "R05": ("Negative", "Cap and weight are problems."),
        "R06": ("Neutral", "Positive value but difficult cleaning."),
        "R07": ("Neutral", "Good cooling but poor paint durability."),
        "R08": ("Neutral", "An indifferent opinion."),
        "R09": ("Negative", "Insulation did not meet expectations."),
        "R10": ("Positive", "Useful and easy to wash."),
        "R11": ("Neutral", "Mixed opinion about capacity."),
        "R12": ("Positive", "Strong positive feedback."),
        "R13": ("Negative", "Broken lid and poor quality."),
        "R14": ("Neutral", "Mostly factual description."),
        "R15": ("Negative", "Poor cooling performance."),
        "R16": ("Positive", "Affordable and practical."),
        "R17": ("Neutral", "Acceptable but uncomfortable."),
        "R18": ("Positive", "Customer expresses happiness.")
    }

    audit_groups = []

    for category in sentiment_order:

        group = df[
            df["Sentiment"] == category
        ].copy()

        if category == "Positive":
            group = group.sort_values(
                "Compound Score",
                ascending=False
            )

        elif category == "Negative":
            group = group.sort_values(
                "Compound Score",
                ascending=True
            )

        else:
            group["Distance"] = (
                group["Compound Score"].abs()
            )

            group = group.sort_values("Distance")

        audit_groups.append(group.head(2))

    audit_df = pd.concat(
        audit_groups,
        ignore_index=True
    )

    audit_df["Human Sentiment"] = audit_df["ID"].apply(
        lambda x: human_review[x][0]
    )

    audit_df["Assessment"] = audit_df.apply(
        lambda row: (
            "Reasonable"
            if row["Sentiment"] == row["Human Sentiment"]
            else "Needs Review"
        ),
        axis=1
    )

    audit_df["Explanation"] = audit_df["ID"].apply(
        lambda x: human_review[x][1]
    )

    st.dataframe(
        audit_df[
            [
                "ID",
                "Comment",
                "Sentiment",
                "Compound Score",
                "Human Sentiment",
                "Assessment",
                "Explanation"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    reasonable = (
        audit_df["Assessment"] == "Reasonable"
    ).sum()

    st.metric(
        "Reasonable Labels",
        f"{reasonable} / {len(audit_df)}"
    )

    st.write(
        "The audit compares automated classifications "
        "with human interpretations. Disagreements "
        "should be reviewed because sentiment analysis "
        "can oversimplify mixed opinions."
    )

    # ======================================
    # NEGATION TEST
    # ======================================

    st.divider()
    st.subheader("Negation Test")

    test_comments = [
        "The bottle is easy to clean.",
        "The bottle is not easy to clean."
    ]

    test_results = []

    for comment in test_comments:

        label, score = analyze_sentiment(comment)

        test_results.append({
            "Comment": comment,
            "Sentiment": label,
            "Compound Score": score
        })

    st.dataframe(
        pd.DataFrame(test_results),
        use_container_width=True,
        hide_index=True
    )

    st.write(
        "The word 'not' changes the meaning of the "
        "sentence from positive to negative. Comparing "
        "the compound scores helps evaluate how VADER "
        "responds to negation. However, automated "
        "analysis may still misunderstand sarcasm, "
        "context, or complicated expressions."
    )

# ==========================================
# TAB 4: TOPIC ANALYSIS
# ==========================================

with tab4:

    st.header("Sentiment Analysis by Topic")

    st.write(
        "Each review is assigned one main topic, "
        "such as insulation, cleaning, or lid usability."
    )

    if filtered_df.empty:

        st.warning("No data available.")

    else:

        topic_counts = pd.crosstab(
            filtered_df["Topic"],
            filtered_df["Sentiment"]
        )

        topic_counts = topic_counts.reindex(
            columns=sentiment_order,
            fill_value=0
        )

        topic_counts["Total"] = (
            topic_counts.sum(axis=1)
        )

        topic_counts = topic_counts.sort_values(
            "Total",
            ascending=False
        )

        st.subheader("Sentiment Count by Topic")

        st.dataframe(
            topic_counts,
            use_container_width=True
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        left_values = pd.Series(
            0,
            index=topic_counts.index,
            dtype=float
        )

        for sentiment in sentiment_order:

            ax.barh(
                topic_counts.index,
                topic_counts[sentiment],
                left=left_values,
                label=sentiment,
                color=colors[sentiment]
            )

            left_values += topic_counts[sentiment]

        ax.invert_yaxis()
        ax.set_xlabel("Number of Reviews")
        ax.set_ylabel("Main Topic")
        ax.set_title("Sentiment Distribution by Topic")
        ax.legend(title="Sentiment")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.info(
            "This comparison is exploratory. The dataset "
            "contains only 18 synthetic comments, and "
            "some topics have very few reviews. Therefore, "
            "the results should not be used to make "
            "general conclusions about customer satisfaction."
        )

# ==========================================
# TAB 5: ASSIGNMENT REPORT
# ==========================================

with tab5:

    st.header("Lab 8 Assignment Report")

    full_counts = (
        df["Sentiment"]
        .value_counts()
        .reindex(sentiment_order, fill_value=0)
    )

    largest_group = full_counts.idxmax()

    # ======================================
    # TASK 1: RUN AND DESCRIBE
    # ======================================

    st.subheader("1. Run and Describe")

    st.write(
        f"The dashboard successfully analyzes {len(df)} "
        f"product reviews. The sentiment analysis identifies "
        f"{full_counts['Positive']} positive, "
        f"{full_counts['Neutral']} neutral, and "
        f"{full_counts['Negative']} negative comments. "
        f"The largest sentiment group is {largest_group.lower()}."
    )

    st.write(
        "A recurring theme is product usability, particularly "
        "the lid, insulation, and cleaning. The mixed-tone "
        "comment R06 shows that positive and negative opinions "
        "can appear within the same response."
    )

    # ======================================
    # TASK 2: QUALITATIVE VIEW
    # ======================================

    st.subheader("2. Improve the Qualitative View")

    st.write(
        "The word 'bottle' was removed from the original "
        "custom stop-word list to observe how frequent "
        "generic terms affect the word cloud. The comparison "
        "option demonstrates that excluding such terms "
        "makes other product features easier to notice."
    )

    st.write(
        "The useful term 'lid' was retained because it "
        "highlights a product feature associated with "
        "customer complaints and feedback."
    )

    st.caption(
        "Word size represents frequency, not importance, "
        "customer satisfaction, or sentiment."
    )

    # ======================================
    # TASK 3: AUDIT SENTIMENT LABELS
    # ======================================

    st.subheader("3. Audit Sentiment Labels")

    st.write(
        "Six comments were selected for manual review, "
        "consisting of two comments from each sentiment "
        "category. Human judgments were compared with "
        "the automated classifications."
    )

    st.write(
        "The negation test compares 'The bottle is easy "
        "to clean' with 'The bottle is not easy to clean.' "
        "This helps determine whether the model recognizes "
        "how negation changes the meaning of a sentence."
    )

    # ======================================
    # TASK 4: COMPARE BY TOPIC
    # ======================================

    st.subheader("4. Compare and Communicate")

    st.write(
        "Each comment is assigned a main topic, including "
        "insulation, cleaning, lid usability, and price. "
        "The stacked bar chart compares sentiment counts "
        "across these topics."
    )

    st.write(
        "However, the sample is too small to support "
        "reliable comparisons between product features."
    )

    # ======================================
    # 150-200 WORD PRODUCT TEAM SUMMARY
    # ======================================

    st.subheader("Product Team Summary (150–200 Words)")

    summary = (
        f"This analysis reviewed 18 synthetic customer comments "
        f"about a reusable water bottle. VADER assigned "
        f"{full_counts['Positive']} positive, "
        f"{full_counts['Neutral']} neutral, and "
        f"{full_counts['Negative']} negative labels, making "
        f"{largest_group.lower()} the largest sentiment group. "
        "The comments suggest that customers notice practical "
        "features such as insulation, the lid, portability, and "
        "cleaning. Several reviews describe convenient daily use, "
        "while others raise concerns about leaking, lid durability, "
        "and ease of maintenance. These are useful starting points "
        "for product improvement. However, the automated labels "
        "should not be treated as confirmed customer opinions. "
        "For example, a customer can appreciate the price while "
        "also finding the bottle difficult to clean. A single "
        "category hides that mixed experience. The word cloud "
        "also shows repeated words rather than how strongly "
        "customers feel about them. Importantly, this dataset "
        "contains only 18 invented reviews, so it cannot "
        "represent real buyers or support reliable topic rankings. "
        "The product team should collect a larger sample of genuine "
        "feedback from customers with different usage patterns, "
        "ask focused follow-up questions about lid performance "
        "and cleaning, and manually validate a balanced subset "
        "of sentiment labels before making decisions."
    )

    st.write(summary)

    st.caption(
        f"Word count: {len(summary.split())} words"
    )

    # ======================================
    # REFLECTION
    # ======================================

    st.subheader("Reflection")

    st.write(
        "The sentiment chart helps identify the overall "
        "distribution of positive, neutral, and negative "
        "feedback. The word cloud highlights frequently "
        "mentioned product features, while the topic chart "
        "shows how sentiment varies across product aspects."
    )

    st.write(
        "However, these visualizations can be misleading "
        "because word frequency does not indicate sentiment "
        "strength, and a single sentiment label may hide "
        "mixed opinions. The small synthetic dataset also "
        "limits the reliability of the findings."
    )

    st.write(
        "To improve validity, the automated classifications "
        "should be compared with human judgments. A larger "
        "and more representative collection of real customer "
        "feedback should also be gathered before making "
        "product development decisions."
    )

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "TFB3133 / TEB3133 Data Visualization | "
    "Lab 8 - Qualitative Data and Sentiment Analysis"
)
