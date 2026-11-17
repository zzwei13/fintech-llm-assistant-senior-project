"""
向量化與檢索（ingest 階段 4）

  bge-m3（Ollama）把新聞片段轉成 1024 維向量 → 存進 Chroma（本機資料夾 chroma_db/）
  之後 RAG 檢索時，用同一個模型把查詢轉成向量，找出餘弦相似度最高的片段。

切塊（chunk）：LangChain RecursiveCharacterTextSplitter
  依「段落 → 句子 → 子句 → 字元」的順序切成最多 CHUNK_CHARS 字的片段，前後重疊 CHUNK_OVERLAP 字；
  每段前面加上標題，
  讓片段單獨被檢索到時仍知道是哪篇新聞在講什麼。
  只索引 relevance 為 main / mention 的新聞（unrelated 不進向量庫）。

metadata（檢索時可過濾）：
  news_id   對回 MySQL 取情緒、理由等最新欄位
  stock_id  只檢索某檔股票
  date      整數 YYYYMMDD，Chroma 的過濾只支援數字比較，用來限定日期範圍
  relevance main / mention
  source, title, url

ID：md5(stock_id + url) + 片段序號 → 重跑時已存在的片段會略過，不會重複寫入

單獨測試（需先跑過 python ingest.py index）：
    python vectorstore.py --stock 2330 --query "法說會 營收展望"
"""

import argparse
import hashlib
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
COLLECTION = "news"
CHUNK_CHARS = 400  # 每個片段最多幾個字
CHUNK_OVERLAP = 50  # 前後片段重疊幾個字，避免重點剛好被切在邊界

_store = None
_splitter = None


def get_splitter():
    """
    依序嘗試分隔符號：段落 → 句子 → 子句 → 字元，盡量不把句子切斷。
    預設分隔符號是給英文的（空白、句點），中文要自己指定。
    keep_separator="end"：句號留在句尾，而不是跑到下一段開頭。
    """
    global _splitter
    if _splitter is None:
        _splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_CHARS,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n", "。", "！", "？", "；", "，", ""],
            keep_separator="end",
        )
    return _splitter


def get_store():
    """第一次呼叫時才建立；Chroma 會自動讀取或建立 chroma_db/ 資料夾。"""
    global _store
    if _store is None:
        _store = Chroma(
            collection_name=COLLECTION,
            embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
            persist_directory=PERSIST_DIR,
            collection_metadata={"hnsw:space": "cosine"},  # 用餘弦距離（預設是 L2）
        )
    return _store


def split_chunks(content):
    """切成約 CHUNK_CHARS 字的片段（LangChain RecursiveCharacterTextSplitter）。"""
    return get_splitter().split_text(content)


def news_to_documents(row):
    """一篇新聞 → 多個 Document（片段 + metadata）與對應的 ID。"""
    base_id = hashlib.md5(f"{row['stock_id']}{row['url']}".encode()).hexdigest()
    metadata = {
        "news_id": row["id"],
        "stock_id": row["stock_id"],
        "date": int(row["date"].strftime("%Y%m%d")),
        "relevance": row["relevance"],
        "source": row["source"],
        "title": row["title"],
        "url": row["url"],
    }
    docs, ids = [], []
    for i, chunk in enumerate(split_chunks(row["content"])):
        docs.append(
            Document(page_content=f"標題：{row['title']}\n{chunk}", metadata=metadata)
        )
        ids.append(f"{base_id}-{i}")
    return docs, ids


def existing_ids():
    """向量庫中已存在的片段 ID。"""
    return set(get_store().get(include=[])["ids"])


def add_news(rows, batch_size=64):
    """把新聞寫入向量庫，已存在的片段略過；回傳 (新增新聞篇數, 新增片段數)。"""
    have = existing_ids()
    new_docs, new_ids, n_news = [], [], 0
    for row in rows:
        docs, ids = news_to_documents(row)
        pending = [(d, i) for d, i in zip(docs, ids) if i not in have]
        if pending:
            n_news += 1
            new_docs += [d for d, _ in pending]
            new_ids += [i for _, i in pending]

    store = get_store()
    for i in range(0, len(new_docs), batch_size):  # 分批送進 bge-m3，避免一次太多
        store.add_documents(
            new_docs[i : i + batch_size], ids=new_ids[i : i + batch_size]
        )
    return n_news, len(new_docs)


def sync_relevance(stock_id, relevance_by_id):
    """
    重新判斷相關性後，讓向量庫的 metadata 跟 MySQL 一致（不重新向量化）：
      變成 unrelated／沒有結果 → 刪除片段
      main ↔ mention 互換      → 只更新 metadata 的 relevance
    原本是 unrelated、現在變成 main / mention 的新聞不在向量庫裡，之後由 add_news 補上。
    回傳 (刪除片段數, 更新片段數)
    """
    store = get_store()
    got = store.get(where={"stock_id": stock_id}, include=["metadatas"])
    to_delete, upd_ids, upd_meta = [], [], []
    for cid, m in zip(got["ids"], got["metadatas"]):
        rel = relevance_by_id.get(m["news_id"])
        if rel not in ("main", "mention"):
            to_delete.append(cid)
        elif rel != m["relevance"]:
            upd_ids.append(cid)
            upd_meta.append({**m, "relevance": rel})
    if to_delete:
        store.delete(ids=to_delete)
    if upd_ids:
        store._collection.update(
            ids=upd_ids, metadatas=upd_meta
        )  # 只改 metadata，向量不變
    return len(to_delete), len(upd_ids)


def search(
    query,
    stock_id,
    k=8,
    start_date=None,
    end_date=None,
    relevance=("main", "mention"),
    news_ids=None,
):
    """
    檢索某檔股票最相關的片段，回傳 [(Document, 距離)]，距離越小越相似（餘弦距離 = 1 − 相似度）。
    start_date / end_date：整數 YYYYMMDD
    news_ids：只在這些新聞中檢索（chain.py 在事件內找證據片段用）
    """
    conditions = [{"stock_id": stock_id}, {"relevance": {"$in": list(relevance)}}]
    if news_ids:
        conditions.append({"news_id": {"$in": list(news_ids)}})
    if start_date:
        conditions.append({"date": {"$gte": start_date}})
    if end_date:
        conditions.append({"date": {"$lte": end_date}})
    return get_store().similarity_search_with_score(
        query, k=k, filter={"$and": conditions}
    )


def news_text(news_id):
    """一篇新聞在向量庫中的所有片段接起來（標題＋內文），供數字核對用"""
    docs = get_store().get(where={"news_id": news_id}, include=["documents"])[
        "documents"
    ]
    return "\n".join(docs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="向量檢索測試")
    parser.add_argument("--stock", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    print(f"向量庫共 {len(existing_ids())} 個片段\n")
    for doc, dist in search(args.query, args.stock, k=args.k):
        m = doc.metadata
        body = doc.page_content.split("\n", 1)[1][:80]
        print(
            f"相似度 {1 - dist:.3f}｜{m['date']}｜{m['relevance']:<7}｜{m['title'][:30]}\n    {body}\n"
        )
