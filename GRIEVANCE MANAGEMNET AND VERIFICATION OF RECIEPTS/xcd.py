import streamlit as st
import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain_community.llms import Together

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Grievance Classifier", layout="wide")
st.title("📋 Grievance Classification using RAG + LLM")

# Upload CSV
grievance_file = st.file_uploader("Upload grievance CSV", type="csv")

# Load context file
file_path = Path("delivery_context_detailed.txt")
if not file_path.exists():
    st.error("delivery_context_detailed.txt not found in the current directory.")
    st.stop()

loader = TextLoader(file_path=str(file_path), encoding="utf-8")
documents = loader.load()

# Create retriever (FAISS only)
text_splitter = CharacterTextSplitter(chunk_size=256, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
faiss_vectorstore = FAISS.from_documents(texts, embedding)
retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 5})

# Prompt
prompt_template = PromptTemplate(
    input_variables=["question", "context"],
    template="""
You are a smart assistant trained to classify customer grievances into one of the predefined categories below. 
Read the grievance carefully and identify the most suitable category based solely on the definitions and examples provided in the context.

Rules:
- Only choose from the listed categories.
- Do not invent new categories.
- If the grievance mentions multiple issues, select the category that is **most central or critical**.
- Be concise but specific in your explanation.

Grievance:
{question}

Context:
{context}

Instructions:
- First, understand the grievance.
- Second, compare it against the category definitions and examples.
- Third, select the best matching category.
- Respond only in the format: <category>: <brief explanation>

Make sure your output is structured exactly like this:
delivery: The grievance describes a delivery delay which fits the delivery category.
"""
)

# LLM setup
os.environ["TOGETHER_API_KEY"] = "tgp_v1__WHfd695e18X5A0K1mLhn-esUWr7lPUEBj2fAgQWbew"

llm = Together(
    model="meta-llama/Llama-3-70b-chat-hf",
    temperature=0.0,
    max_tokens=512,
    together_api_key=os.environ["TOGETHER_API_KEY"],
)

chain = LLMChain(llm=llm, prompt=prompt_template)

VALID_CATEGORIES = ["billing", "service", "delivery", "technical","behaviour"]

# ----------------------- CSV Classification -----------------------
if grievance_file:
    df = pd.read_csv(grievance_file)
    cust_ids = df["customer_id"].tolist()
    selected_id = st.selectbox("Select a Customer ID", cust_ids)

    grievance_row = df[df["customer_id"] == selected_id].iloc[0]
    grievance = grievance_row["grievance"]

    with st.spinner("Classifying grievance from CSV..."):
        retrieved_docs = retriever.get_relevant_documents(grievance)
        top_context = "\n".join([doc.page_content for doc in retrieved_docs[:3]])

        response = chain.run({"question": grievance, "context": top_context})

        if ":" in response:
            label, explanation = response.split(":", 1)
        else:
            label, explanation = response, "No explanation provided."

        label = label.strip().lower()
        explanation = explanation.strip()

        

        st.markdown("### 📝 Grievance")
        st.write(grievance)

        

        st.markdown("### 💡 Explanation")
        st.info(explanation)

        st.markdown("### 📚 Context Used")
        st.code(top_context, language="markdown")

# -------------------- Manual Classification --------------------
st.markdown("---")
st.subheader("✍️ Manually Enter a Grievance")

manual_input = st.text_area("Enter a customer grievance to classify it manually:")

if st.button("Classify Manually Entered Grievance") and manual_input.strip():
    grievance = manual_input.strip()

    with st.spinner("Classifying manual grievance..."):
        retrieved_docs = retriever.get_relevant_documents(grievance)
        top_context = "\n".join([doc.page_content for doc in retrieved_docs[:3]])

        response = chain.run({"question": grievance, "context": top_context})

        st.markdown("### 🧪 Raw Model Response")
        st.code(response, language="text")

        if ":" in response:
            label, explanation = response.split(":", 1)
        else:
            label, explanation = response, "No explanation provided."

        label = label.strip().lower()
        explanation = explanation.strip()

        
        st.markdown("### 📝 Grievance")
        st.write(grievance)

    
        st.success(label)

        st.markdown("### 💡 Explanation")
        st.info(explanation)

        st.markdown("### 📚 Context Used")
        st.code(top_context, language="markdown")
