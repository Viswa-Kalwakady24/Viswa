# app.py
import streamlit as st
from model_pipeline import query_model
from utils.budget_tools import generate_budget_summary
from utils.financial_insights import spending_insights
from utils.tone_adapter import adapt_tone

# Page config
st.set_page_config(page_title="Personal Finance Chatbot", page_icon="💰", layout="wide")

# Header
with st.container():
    st.title("💰 Personal Finance Chatbot (IBM Granite 3.3 2B Instruct)")
    st.write("Ask questions about savings, taxes, investments, and get budget summaries and spending insights.")

# Sidebar: user profile & inputs
st.sidebar.header("User Profile")
user_type = st.sidebar.selectbox("I am a:", ["Student", "Professional"], index=1)

st.sidebar.markdown("---")
st.sidebar.header("Enter up to 6 expense items")
expense_count = st.sidebar.selectbox("How many expense lines?", [1, 2, 3, 4, 5, 6], index=2)

# Collect expenses
expenses = []
for i in range(expense_count):
    cat_key = f"cat_{i}"
    amt_key = f"amt_{i}"
    category = st.sidebar.text_input(f"Category {i+1}", key=cat_key)
    amount = st.sidebar.number_input(f"Amount {i+1}", min_value=0.0, step=1.0, key=amt_key)
    if category:
        expenses.append((category.strip(), float(amount)))

st.sidebar.markdown("---")
st.sidebar.info(
    "Using IBM Granite 3.3 2B Instruct locally. Ensure the model is downloaded or configured."
)

# Main area
st.header("🧠 Ask a financial question")
user_query = st.text_input("Type your question (e.g., 'How can I save more this month?')", key="user_query")

col1, col2 = st.columns(2)

with col1:
    if st.button("Get Guidance", key="btn_guidance"):
        if not user_query or not user_query.strip():
            st.warning("Please enter a question.")
        else:
            prompt = user_query.strip()

            # Include expense context
            if expenses:
                prompt += "\n\nUser expenses:\n" + "\n".join([f"- {c}: {a:.2f}" for c, a in expenses])

            raw_response = query_model(prompt)  # <-- Granite model will answer here
            final_response = adapt_tone(user_type, raw_response)
            st.success(final_response)

with col2:
    st.write("Quick actions")
    if st.button("Generate Budget Summary", key="btn_summary"):
        if not expenses:
            st.warning("Please enter at least one expense in the sidebar.")
        else:
            df = generate_budget_summary(expenses)
            st.dataframe(df)

    if st.button("Show Spending Insights", key="btn_insights"):
        insights = spending_insights(expenses)
        for insight in insights:
            st.info(insight)

st.markdown("---")
st.write("⚙ Developer notes:")
st.write("- This version uses IBM Granite 3.3 2B Instruct inside model_pipeline.py.")
st.write("- Ensure granite checkpoint is installed or downloaded.")