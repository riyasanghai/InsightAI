
import streamlit as st
import pandas as pd
import numpy as np
from google import genai
import os

# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="InsightAI",
    page_icon="📊",
    layout="wide"
)

st.title("📊 InsightAI")
st.subheader("Business Analytics Agent")

st.write(
    "Ask natural-language questions about business data "
    "and receive AI-powered business insights."
)

st.divider()


# ==========================================
# GEMINI SETUP
# ==========================================

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    st.error("Gemini API key not found. Load it from Colab Secrets first.")
    st.stop()

client = genai.Client(api_key=api_key)


# ==========================================
# RAG KNOWLEDGE BASE
# ==========================================

knowledge_text = """
1. Sales Value
Sales value is calculated as the sum of product prices across order items.
Freight charges are not included in this sales value metric.

2. Average Order Value
AOV is calculated by summing product prices for each order
and then taking the average across orders.

3. Late Delivery
An order is late when the actual delivery date is after
the estimated delivery date.

4. Delivery Analysis
Orders without an actual delivery date are excluded from
delivery-time calculations.

5. Review Score
Customer review scores range from 1 to 5.
A score of 1 is very poor, while 5 is excellent.

6. High-Value Customer
A high-value customer is defined in this project as a customer
whose total purchase value is in the top 25 percent of customers.

7. Order-Level Analysis
An order may contain multiple products, so order-level metrics
must be calculated at the order level to avoid double counting.

8. Product Categories
Product categories are the translated English category names
used in the Olist dataset.

9. Business Insight
Business insights should be supported by dataset evidence
or relevant information from this knowledge base.

10. Data Limitations
The Olist dataset contains historical Brazilian e-commerce data.
It is not real-time business data.
"""

knowledge_chunks = [
    chunk.strip()
    for chunk in knowledge_text.split("\n\n")
    if chunk.strip()
]


# ==========================================
# RAG EMBEDDINGS
# ==========================================

@st.cache_data
def get_knowledge_embeddings(chunks, key):
    embedding_client = genai.Client(api_key=key)
    embeddings = []

    for chunk in chunks:
        result = embedding_client.models.embed_content(
            model="gemini-embedding-001",
            contents=chunk
        )
        embeddings.append(result.embeddings[0].values)

    return embeddings


def create_embedding(text):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values


# ==========================================
# RAG RETRIEVAL
# ==========================================

def retrieve_knowledge(question, top_k=2):
    knowledge_embeddings = get_knowledge_embeddings(
        knowledge_chunks,
        api_key
    )

    question_embedding = np.array(create_embedding(question))
    similarities = []

    for i, chunk_embedding in enumerate(knowledge_embeddings):
        chunk_embedding = np.array(chunk_embedding)

        denominator = (
            np.linalg.norm(question_embedding)
            * np.linalg.norm(chunk_embedding)
        )

        if denominator == 0:
            score = 0
        else:
            score = np.dot(
                question_embedding,
                chunk_embedding
            ) / denominator

        similarities.append((i, score))

    similarities.sort(
        key=lambda item: item[1],
        reverse=True
    )

    return [
        knowledge_chunks[i]
        for i, score in similarities[:top_k]
    ]


def generate_rag_answer(question):
    retrieved = retrieve_knowledge(question)
    context = "\n\n".join(retrieved)

    prompt = f"""
You are an AI business analytics assistant.

Answer the user's question using the retrieved knowledge.

Retrieved Knowledge:
{context}

User Question:
{question}

Rules:
- Use the retrieved knowledge as your source.
- Do not invent facts.
- Keep the answer clear and simple.
- If the knowledge does not contain enough information,
  say that the information is not available.
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text, retrieved


# ==========================================
# ANALYTICS FUNCTIONS
# ==========================================

def get_category_sales(data):
    if (
        "category" not in data.columns
        or "price" not in data.columns
    ):
        return "This dataset does not contain category and price columns."

    return (
        data.groupby("category")["price"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )


def get_state_sales(data):
    if (
        "customer_state" not in data.columns
        or "price" not in data.columns
    ):
        return "This dataset does not contain state and price columns."

    return (
        data.groupby("customer_state")["price"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )


def get_average_order_value(data):
    if (
        "order_id" not in data.columns
        or "price" not in data.columns
    ):
        return "This dataset does not contain order_id and price columns."

    order_values = data.groupby("order_id")["price"].sum()

    return round(order_values.mean(), 2)


def get_business_summary(data):
    summary = {}

    if "order_id" in data.columns:
        summary["total_orders"] = data["order_id"].nunique()

    if "customer_unique_id" in data.columns:
        summary["total_customers"] = (
            data["customer_unique_id"].nunique()
        )

    if "price" in data.columns:
        summary["total_sales"] = round(data["price"].sum(), 2)

    return summary


# ==========================================
# AUTOMATIC CHART
# ==========================================

def display_chart(result):
    if not isinstance(result, pd.Series):
        return

    if result.empty:
        return

    st.subheader("📊 Data Visualization")

    chart_data = result.sort_values(ascending=True)

    st.bar_chart(
        chart_data,
        horizontal=True
    )


# ==========================================
# AGENT ROUTER
# ==========================================

def choose_action(question):
    prompt = f"""
You are an AI business analytics agent.

Choose ONE action to answer the user's question.

Available actions:

category_sales
- Sales by product category.

state_sales
- Sales by state or region.

average_order_value
- Average value of an order.

business_summary
- Overall metrics such as orders, customers and total sales.

rag
- Definitions, business rules, terminology and information
  in the business knowledge base.

Rules:
- Choose an analytics action when calculations from the
  uploaded dataset are needed.
- Choose rag for questions answerable from the knowledge base.
- Return ONLY one action name from the list.
- Do not explain your choice.

User question:
{question}
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    action = response.text.strip().lower()

    # Handle accidental Markdown formatting from the model.
    action = action.replace("`", "").strip()

    return action


# ==========================================
# DATA UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload your business dataset",
    type=["csv"]
)

if uploaded_file is not None:

    try:
      try:
        analysis_data = pd.read_csv(uploaded_file, encoding="utf-8")
        
      except UnicodeDecodeError:
        uploaded_file.seek(0)
        analysis_data = pd.read_csv(uploaded_file, encoding="latin1")

    except Exception as e:
        st.error(f"Could not read the CSV file: {e}")
        st.stop()

    # Standardize column names and recognize common dataset variants.
    analysis_data.columns = (
        analysis_data.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    # Map common column names to InsightAI's standard names.
    # Existing canonical columns are kept if they are already present.
    column_aliases = {
        "sales": "price",
        "customer_id": "customer_unique_id",
        "region": "customer_state",
    }

    for source_column, standard_column in column_aliases.items():
        if source_column in analysis_data.columns and standard_column not in analysis_data.columns:
            analysis_data.rename(
                columns={source_column: standard_column},
                inplace=True
            )

    st.success(
        f"Dataset loaded successfully: "
        f"{analysis_data.shape[0]:,} rows × "
        f"{analysis_data.shape[1]} columns"
    )


    # ==========================================
    # KPI DASHBOARD
    # ==========================================

    st.subheader("📈 Business Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        if "order_id" in analysis_data.columns:
            st.metric(
                "Total Orders",
                f"{analysis_data['order_id'].nunique():,}"
            )
        else:
            st.metric("Total Orders", "N/A")

    with col2:
        if "customer_unique_id" in analysis_data.columns:
            st.metric(
                "Unique Customers",
                f"{analysis_data['customer_unique_id'].nunique():,}"
            )
        else:
            st.metric("Unique Customers", "N/A")

    with col3:
        if "price" in analysis_data.columns:
            st.metric(
                "Total Sales",
                f"{analysis_data['price'].sum():,.2f}"
            )
        else:
            st.metric("Total Sales", "N/A")


    with st.expander("Preview dataset"):
        st.dataframe(
            analysis_data.head(10),
            use_container_width=True
        )

    st.divider()

    # ==========================================
    # QUESTION INPUT
    # ==========================================

    question = st.text_input(
        "Ask a business question:",
        placeholder="Example: Which category generated the highest sales?"
    )

    # ==========================================
    # TOOL REGISTRY
    # ==========================================

    tools = {
        "category_sales": lambda: get_category_sales(analysis_data),
        "state_sales": lambda: get_state_sales(analysis_data),
        "average_order_value": lambda: get_average_order_value(analysis_data),
        "business_summary": lambda: get_business_summary(analysis_data)
    }

    # ==========================================
    # ANALYZE QUESTION
    # ==========================================

    if st.button("🔍 Analyze"):

        if not question.strip():
            st.warning("Please enter a business question.")

        else:
            with st.spinner("Analyzing your question..."):

                try:
                    action = choose_action(question)

                    if action == "rag":

                        answer, retrieved = generate_rag_answer(question)

                        st.subheader("💡 Business Insight")
                        st.write(answer)

                        with st.expander("🔎 Show analysis details"):
                            st.write("Selected analysis: RAG")
                            st.write("Retrieved knowledge:")

                            for chunk in retrieved:
                                st.write(chunk)

                    elif action in tools:

                        result = tools[action]()

                        # Display a chart when the result is chartable.
                        display_chart(result)

                        prompt = f"""
You are an AI business analytics assistant.

User question:
{question}

Selected analysis:
{action}

Analytics result:
{result}

Answer using ONLY the provided analytics result.
Do not invent numbers.
Explain the result in simple business language.
If the result says required columns are missing,
explain that limitation instead of inventing an answer.
"""

                        response = client.models.generate_content(
                            model="gemini-3.5-flash-lite",
                            contents=prompt
                        )

                        st.subheader("💡 Business Insight")
                        st.write(response.text)

                        with st.expander("🔎 Show analysis details"):
                            st.write("Selected analysis:", action)
                            st.write("Data used:")
                            st.write(result)

                    else:
                        st.warning(
                            "The agent returned an unrecognised action. "
                            "Please try rephrasing your question."
                        )

                except Exception as e:
                    st.error(
                        "Something went wrong while analyzing your "
                        "question. Check your API key, model access, "
                        "and connection, then try again."
                    )
                    st.caption(f"Technical details: {e}")

else:
    st.info("Upload a CSV dataset to start asking business questions.")