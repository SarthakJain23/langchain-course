import asyncio
import os
from typing import List

import certifi
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

REQUIRED_ENV_VARS = [
    "LANGCHAIN_DOC_INDEX_NAME",
    "PINECONE_API_KEY",
    "GOOGLE_API_KEY",
    "TAVILY_API_KEY",
]


def validate_env() -> None:
    """Validate that all required environment variables are set."""
    missing = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}"
        )


def get_vectorstore() -> PineconeVectorStore:
    """Initialize vectorstore client."""
    index_name = os.getenv("LANGCHAIN_DOC_INDEX_NAME")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        output_dimensionality=1536,
    )
    return PineconeVectorStore(index_name=index_name, embedding=embeddings)


async def index_documents_async(
    vectorstore: PineconeVectorStore,
    docs: List[Document],
    batch_size: int = 50,
) -> None:
    """Index documents in batches sequentially to prevent aiohttp session closure issues."""
    if not docs:
        print("⚠️ No documents to index.")
        return

    batches = [docs[i : i + batch_size] for i in range(0, len(docs), batch_size)]
    successful = 0
    failed = 0

    for i, batch in enumerate(batches, 1):
        try:
            ids = [doc.metadata.get("id") for doc in batch]
            if all(ids):
                await vectorstore.aadd_documents(batch, ids=ids)
            else:
                await vectorstore.aadd_documents(batch)

            print(f"✅ Indexed batch {i}/{len(batches)} ({len(batch)} chunks)")
            successful += 1
        except Exception as e:
            print(f"❌ Failed to index batch {i}: {e}")
            failed += 1

    print(
        f"\n📊 Indexing Summary: {successful} successful batches, {failed} failed batches."
    )


async def main():
    validate_env()
    print("🚀 Starting documentation crawling...")

    tavily_crawl = TavilyCrawl()

    res = tavily_crawl.invoke(
        {
            "url": "https://docs.langchain.com/oss/python/langchain/overview",
            "max_depth": 5,
            "extract_depth": "advanced",
        }
    )

    results_list = res.get("results", [])
    print(f"📄 Crawled {len(results_list)} pages.")

    all_docs: List[Document] = []
    for result in results_list:
        content = result.get("raw_content")
        url = result.get("url")

        if content and content.strip():
            all_docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": url,
                        "title": result.get("title", ""),
                    },
                )
            )

    print(f"🧹 Valid documents with content: {len(all_docs)}")

    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=1000,
        chunk_overlap=200,
    )
    splitted_docs = text_splitter.split_documents(all_docs)

    for i, doc in enumerate(splitted_docs):
        doc.metadata["chunk_index"] = i
        doc.metadata["id"] = f"{doc.metadata.get('source')}#chunk-{i}"

    print(f"✂️ Total Markdown chunks created: {len(splitted_docs)}")

    vectorstore = get_vectorstore()
    await index_documents_async(
        vectorstore=vectorstore,
        docs=splitted_docs,
        batch_size=50,
    )


if __name__ == "__main__":
    asyncio.run(main())
