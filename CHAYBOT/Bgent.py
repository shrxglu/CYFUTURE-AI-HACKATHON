# === Standard Imports ===
import os
import re
import sqlite3
import datetime
from random import choice, randint
import streamlit as st
from pydantic import BaseModel

# === LangChain Imports ===
from langchain_community.llms import Together
from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.exceptions import OutputParserException
from langchain.tools import Tool
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# === Set API Key ===
os.environ["TOGETHER_API_KEY"] = "tgp_v1__WHfd695e18X5A0K1mLhn-esUWr7lPUEBj2fAgQWbew"

# === Memory ===
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# === LLM ===
llm = Together(
    model="google/gemma-2-27b-it",
    temperature=0.0,
    max_tokens=512,
    together_api_key=os.environ["TOGETHER_API_KEY"],
)

# === Database Initialization ===
# === Database Initialization (Simplified) ===
conn = sqlite3.connect("ecommerce.db")
cursor = conn.cursor()

# Drop existing tables
cursor.executescript("""
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS products;

-- Customers Table
CREATE TABLE customers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL
);

-- Products Table
CREATE TABLE products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    price REAL NOT NULL,
    stock INTEGER NOT NULL
);

-- Orders Table
CREATE TABLE orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    status TEXT CHECK(status IN ('pending', 'shipped', 'cancelled')) NOT NULL,
    payment_mode TEXT CHECK(payment_mode IN ('online', 'cod')) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(id),
    FOREIGN KEY (product_id) REFERENCES products(id)
);
""")

# Add sample data
cursor.executemany("INSERT INTO customers (name, email) VALUES (?, ?);", [
    ("Alice", "alice@example.com"),
    ("Bob", "bob@example.com"),
])

cursor.executemany("INSERT INTO products (name, price, stock) VALUES (?, ?, ?);", [
    ("Smartphone", 699.99, 10),
    ("Laptop", 1199.99, 5),
])

cursor.executemany("""
INSERT INTO orders (customer_id, product_id, quantity, status, payment_mode)
VALUES (?, ?, ?, ?, ?)
""", [
    (1, 1, 1, "pending", "online"),
    (2, 2, 1, "shipped", "cod"),
])

conn.commit()
conn.close()

# === Model to Validate Order ID ===
# === Helper Models ===
class OrderIDModel(BaseModel):
    order_id: int

    @classmethod
    def validate_order_id(cls, input_str: str):
        try:
            order_id = int(input_str.strip())
            if order_id <= 0:
                raise ValueError
            return cls(order_id=order_id)
        except Exception:
            raise ValueError("Invalid order ID: must be a positive integer")


# === Safe Decorator ===
def safe_order_tool(func):
    def wrapper(input_str: str) -> str:
        try:
            validated = OrderIDModel.validate_order_id(input_str)
        except ValueError as e:
            raise OutputParserException(f"{str(e)} Please provide a valid order ID.")
        return func(validated.order_id)
    return wrapper


# === Simplified Helper Functions ===

def check_order_status(order_id: int) -> str:
    conn = sqlite3.connect("ecommerce.db")
    cursor = conn.cursor()
    cursor.execute("SELECT status FROM orders WHERE id = ?", (order_id,))
    row = cursor.fetchone()
    conn.close()
    return f"Order {order_id} status: {row[0]}" if row else f"Order {order_id} not found."


def cancel_order(order_id: int) -> str:
    conn = sqlite3.connect("ecommerce.db")
    cursor = conn.cursor()
    cursor.execute("SELECT status FROM orders WHERE id = ?", (order_id,))
    row = cursor.fetchone()
    if not row:
        return f"Order {order_id} not found."
    if row[0] == 'cancelled':
        return f"Order {order_id} is already cancelled."
    cursor.execute("UPDATE orders SET status = 'cancelled' WHERE id = ?", (order_id,))
    conn.commit()
    conn.close()
    return f"Order {order_id} has been cancelled."


def get_order_details(order_id: int) -> str:
    conn = sqlite3.connect("ecommerce.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT o.id, c.name, c.email, p.name, p.price, o.quantity, o.status, o.payment_mode, o.created_at
        FROM orders o
        JOIN customers c ON o.customer_id = c.id
        JOIN products p ON o.product_id = p.id
        WHERE o.id = ?
    """, (order_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return f"Order {order_id} not found."
    
    oid, cname, cemail, pname, price, qty, status, payment_mode, created = row
    total = price * qty
    return (
        f"Order ID: {oid}\n"
        f"Customer: {cname} ({cemail})\n"
        f"Product: {pname} | Qty: {qty} | Total: ₹{total:.2f}\n"
        f"Status: {status} | Payment: {payment_mode}\n"
        f"Ordered On: {created}"
    )


def find_user(query: str) -> str:
    conn = sqlite3.connect("ecommerce.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, email FROM customers WHERE name LIKE ? OR email LIKE ? LIMIT 5",
                   (f"%{query}%", f"%{query}%"))
    rows = cursor.fetchall()
    conn.close()
    return "\n".join([f"{r[0]}: {r[1]} ({r[2]})" for r in rows]) if rows else f"No users found matching '{query}'."


# === Stub Functions for Return/Refund (Optional)
def return_order(order_id: int) -> str:
    return "Return/refund tracking is not enabled in this simplified version."


def check_refund_status(order_id: int) -> str:
    return "Refund tracking is not enabled in this simplified version."


# === Safe Tool Wrappers ===
check_order_status_safe = safe_order_tool(check_order_status)
cancel_order_safe = safe_order_tool(cancel_order)
get_order_details_safe = safe_order_tool(get_order_details)
return_order_safe = safe_order_tool(return_order)
check_refund_status_safe = safe_order_tool(check_refund_status)



# === RAG Setup for FAQ ===
loader = loader = UnstructuredFileLoader(r"D:\vit\FAQ.txt")
 # Adjust path if needed
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
split_docs = splitter.split_documents(docs)

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(split_docs, embedding_model)
retriever = vectorstore.as_retriever()

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False
)

def query_faq(input_str: str) -> str:
    return rag_chain.run(input_str)



# === Define Tools for Agent ===
tools = [
    Tool(name="check_order_status", func=check_order_status_safe, description="Check order status by valid order ID."),
    Tool(name="cancel_order", func=cancel_order_safe, description="Cancel an order by valid order ID."),
    Tool(name="get_order_details", func=get_order_details_safe, description="Get order details by valid order ID."),
    Tool(name="find_user", func=find_user, description="Find user by name or email."),
    Tool(name="return_order", func=return_order_safe, description="(Disabled) Return an order by ID."),
    Tool(name="check_refund_status", func=check_refund_status_safe, description="(Disabled) Check refund status."),
    Tool(name="faq_query", func=query_faq, description="Answer FAQ/T&C questions", return_direct=True)
]

agent_kwargs = {
    "system_message": """
You are a helpful e-commerce assistant.
Always respond in this format:

====================
THOUGHT:
<Your reasoning>

TOOL:
<The tool name — or Final Answer>

INPUT:
<Your tool input or the final message>
====================
"""
}

agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent="chat-zero-shot-react-description",
    memory=memory,
    agent_kwargs=agent_kwargs,
    handle_parsing_errors=True,
    verbose=True,
)

def extract_final_answer(text: str) -> str:
    match = re.search(r"(?i)Action:\s*Final Answer\s*Action Input:\s*(.+)", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()

# === Streamlit UI ===
st.set_page_config(page_title="🛍️ E-commerce Chatbot", page_icon="🛒")
st.title("🛍️ E-commerce Assistant")
st.markdown("Ask about orders, returns, refunds, delivery, or policies.")

if "chat" not in st.session_state:
    st.session_state.chat = []

for msg in st.session_state.chat:
    st.chat_message(msg["role"]).markdown(msg["content"])

user_prompt = st.chat_input("Ask me something...")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat.append({"role": "user", "content": user_prompt})
    try:
        result = agent_executor.invoke({"input": user_prompt})
        raw_output = result.get("output", "")
        final_response = extract_final_answer(raw_output)
    except Exception as e:
        final_response = f"[ERROR] {str(e)}"
    st.chat_message("assistant").markdown(final_response)
    st.session_state.chat.append({"role": "assistant", "content": final_response})
