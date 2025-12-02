import streamlit as st
import nest_asyncio
nest_asyncio.apply()
from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

st.set_page_config(
    page_title="CloudflareGPT",
    page_icon="☁️",
)

st.title("CloudflareGPT")

st.markdown(
    """
    Welcome to CloudflareGPT!
    
    Ask questions about the documentation of:
    - AI Gateway
    - Cloudflare Vectorize
    - Workers AI
    """
)

with st.sidebar:
    api_key = st.text_input("OpenAI API Key")
    st.markdown("---")
    st.markdown("[GitHub Repo](https://github.com/banminseok/siteGpt)")

if not api_key:
    st.warning("Please provide an OpenAI API Key in the sidebar.")
    st.stop()

def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
    )

@st.cache_resource(show_spinner="Loading website...")
def load_website(url, filter_urls, api_key):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=750,
        chunk_overlap=80,
    )
    loader = SitemapLoader(
        url,
        filter_urls=filter_urls,
        parsing_function=parse_page,
    )
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings(
        openai_api_key=api_key,
        chunk_size=600,
    ))
    return vector_store.as_retriever()

# Cloudflare Documentation Sitemap
SITEMAP_URL = "https://developers.cloudflare.com/sitemap.xml"
FILTERS = [
    r"https://developers.cloudflare.com/ai-gateway/.*",
    r"https://developers.cloudflare.com/vectorize/.*",
    r"https://developers.cloudflare.com/workers-ai/.*",
]

try:
    retriever = load_website(SITEMAP_URL, FILTERS, api_key)
except Exception as e:
    st.error(f"Error loading sitemap: {e}")
    st.stop()

# Chat Logic
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

llm = ChatOpenAI(
    temperature=0.1,
    openai_api_key=api_key
)

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 10.

    가장 좋은 답변에만 10점을 주세요. 
    A score of 10 must ONLY be given if the context 
    
    **EXACTLY MATCHES** 
    the requested model name 
    (e.g., 'llama-2-7b-chat-fp16') AND the requested pricing unit 
    (e.g., 'per M input tokens').
    
    만약 Context에 다른 모델의 가격, 다른 단위(예: output tokens), 또는 가격 정보가 부분적으로만 일치하는 경우, 
    점수는 5점 이하(0-5)여야 합니다. 불확실하거나 질문과 무관한 Context는 0점 처리해야 합니다.

    그렇게 판단한 결정적인 근거 및 Context도 함께 답변에 포함시켜주세요.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:

    Question: What is the price per 1M input tokens of the llama-3.1-8b-instruct model?
    Context: @cf/meta/llama-3.1-8b-instruct $0.282 per M input tokens
    Answer: $0.282 per M input tokens.
    Reasoning: The context explicitly states the exact price ($0.282) for the exact model (llama-3.1-8b-instruct) per 1M input tokens.
    Score: 10

    Question: What is the price per 1M input tokens of the llama-3.1-8b-instruct model?
    Context: @cf/meta/llama-1-1b-chat-fp16 $0.001 per M input tokens
    Answer: $0.001 per M input tokens.
    Reasoning: The context provides a price per M input tokens, but for a different model.
    Score: 5
                                            
    Question: How far away is the moon?
    Context: ...
    Answer: The moon is 384,400 km away. 
    Reasoning : ...
    Score: 10
                                                  
    Question: How far away is the sun?
    Context: ...
    Answer: I don't know 
    Reasoning : ...
    Score: 0
                                                  
    Your turn!

    Question: {question}
    """
)

def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]    
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata.get("lastmod", ""),
            }
            for doc in docs
        ],
    }

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            답변에 (Context...) 근거는 빼고 답변해주세요.

            
            If the question is in Korean, please answer in Korea
            
            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)

def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    st.write(answers)
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )

query = st.chat_input("Ask a question about Cloudflare AI products...")

if query:
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
        
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )
            result = chain.invoke(query)
            st.markdown(result.content.replace("$", "\$"))
            st.session_state["messages"].append({"role": "ai", "content": result.content})