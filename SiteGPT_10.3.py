from langchain.document_loaders import AsyncChromiumLoader, SitemapLoader
from langchain.document_transformers import Html2TextTransformer
from fake_useragent import UserAgent
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import sys
import asyncio
import logging

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize a UserAgent object
ua = UserAgent()

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)


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
        .replace("CloseSearch Submit Blog", "")
    )


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    try:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000,
            chunk_overlap=200,
        )
        loader = SitemapLoader(
            url,
            # filter_urls=[
            #     r"^(.*\/blog\/).*",
            # ],
            parsing_function=parse_page,
        )
        loader.requests_per_second = 2
        loader.requests_kwargs = {"verify": False}
        # Set a realistic user agent
        loader.headers = {'User-Agent': ua.random}
        #docs = loader.load()
        docs = loader.load_and_split(text_splitter=splitter)
        logging.debug(f"Loaded documents: {docs}")
        return docs
    except Exception as e:
        logging.error(f"Error loading sitemap: {e}")
        return []

html2text_transformer = Html2TextTransformer()

st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)

with st.sidebar:
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
    )
    st.write("https://www.processon.io/sitemap.xml")
    st.write("https://www.gamsgo.com/sitemap_index.xml")


if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL.")
    else:
        docs = load_website(url.strip())
        if docs:
            st.write(docs)
        else:
            st.error(
                "Failed to load documents from the sitemap. Please check the URL and try again."
            )
    # loader = AsyncChromiumLoader([url])
    # docs = loader.load()
    # transformed = html2text_transformer.transform_documents(docs)
    # st.write(docs)