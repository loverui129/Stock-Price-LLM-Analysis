# app/services/rag.py
"""
非常轻量的 RAG：
- 用 RSS 的 title/summary 当作文本来源（不抓整篇网页，足够演示）
- 用 OpenAI Embeddings + 本地 FAISS 建库
- 检索时返回 {source, url, summary} 作为可引用的证据
"""

import os
import pathlib
from typing import List, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = pathlib.Path("data/faiss")
DATA_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")  # 成本低、够用


def _embedder() -> OpenAIEmbeddings:
    # 读取 OPENAI_API_KEY 环境变量
    return OpenAIEmbeddings(model=EMBED_MODEL)


def _docs_from_headlines(ticker: str, headlines: List[Dict[str, Any]]) -> List[Document]:
    """
    用 title + summary 作为内容，metadata 带上 source/url/ticker/published，便于后续引用。
    """
    docs: List[Document] = []
    for h in headlines:
        title = h.get("title", "") or ""
        summary = h.get("summary", "") or ""
        content = (title + "\n" + summary).strip()
        if not content:
            continue
        meta = {
            "ticker": ticker.upper(),
            "source": h.get("source", "") or "",
            "url": h.get("url", "") or "",
            "published": h.get("published") or h.get("published_at") or "",
            "title": title,
        }
        docs.append(Document(page_content=content, metadata=meta))
    return docs


def _chunk(docs: List[Document]) -> List[Document]:
    # 轻量切分（title+summary )
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    return splitter.split_documents(docs)


def _index_path(ticker: str) -> pathlib.Path:
    return DATA_DIR / f"faiss_{ticker.upper()}"


def index_headlines(ticker: str, headlines: List[Dict[str, Any]]) -> None:
    docs = _docs_from_headlines(ticker, headlines)
    if not docs:
        return
    chunks = _chunk(docs)
    path = _index_path(ticker)

    emb = _embedder()
    if path.exists():
        vs = FAISS.load_local(str(path), emb, allow_dangerous_deserialization=True)
        # 按 url 去重（只示例一种简单做法）
        existing = { (d.metadata or {}).get("url") for d in vs.docstore._dict.values() }
        new_chunks = [d for d in chunks if (d.metadata or {}).get("url") not in existing]
        if not new_chunks:
            return
        vs.add_documents(new_chunks, embedding=emb)
    else:
        vs = FAISS.from_documents(chunks, emb)
    vs.save_local(str(path))


def search_evidences(ticker: str, query: str, k: int = 5, min_score: float = 0.3) -> List[Dict[str, Any]]:
    path = _index_path(ticker)
    if not path.exists():
        return []
    vs = FAISS.load_local(str(path), _embedder(), allow_dangerous_deserialization=True)
    docs_scores = vs.similarity_search_with_score(query, k=k)
    out = []
    for d, score in docs_scores:
        # score 越小越相似（FAISS 的距离），可按需要换成阈值判断
        # 这里用一个简单过滤：如果向量库实现返回的是余弦相似度，就改成 score >= min_score
        md = d.metadata or {}
        out.append({
            "source": md.get("source", "") or "",
            "url": md.get("url", "") or "",
            "summary": d.page_content[:240].replace("\n", " ").strip(),
        })
    return out
