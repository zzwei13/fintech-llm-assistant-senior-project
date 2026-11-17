"""
MySQL 連線與存取

連線設定從 rag_news/.env 讀取：
    DB_HOST=localhost
    DB_PORT=3306
    DB_USER=root
    DB_PASSWORD=你的密碼
    DB_NAME=stock_news
"""

import json
import os

import pymysql
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))


def get_connection():
    """建立 MySQL 連線；呼叫端用 with 區塊確保連線關閉。"""
    return pymysql.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "stock_news"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False,
    )


# ---------- stocks ----------


def get_stock_name(stock_id):
    """查股票名稱；查無此代號時回傳 None。"""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT stock_name FROM stocks WHERE stock_id = %s", (stock_id,))
        row = cur.fetchone()
        return row["stock_name"] if row else None


# ---------- news：寫入 ----------

NEWS_COLUMNS = ("stock_id", "date", "published_at", "source", "title", "url", "content")


def get_existing_urls(stock_id):
    """回傳某檔股票已存在的新聞網址，讓爬蟲略過，不必重抓內文。"""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT url FROM news WHERE stock_id = %s", (stock_id,))
        return {row["url"] for row in cur.fetchall()}


def insert_news(articles):
    """
    寫入新聞基本欄位；網址已存在時略過（id = id 不做任何更新）。
    回傳實際新增的筆數。
    """
    if not articles:
        return 0
    cols = ", ".join(NEWS_COLUMNS)
    placeholders = ", ".join(["%s"] * len(NEWS_COLUMNS))
    sql = (
        f"INSERT INTO news ({cols}) VALUES ({placeholders}) "
        "ON DUPLICATE KEY UPDATE id = id"
    )
    rows = [tuple(a[c] for c in NEWS_COLUMNS) for a in articles]
    with get_connection() as conn:
        with conn.cursor() as cur:
            inserted = 0
            for row in rows:
                inserted += cur.execute(sql, row)  # 新增回傳 1，已存在回傳 0
        conn.commit()
    return inserted


# ---------- news：查詢（供後續階段使用） ----------


def count_news(stock_id=None):
    """依股票統計新聞篇數與各欄位的完成度。"""
    sql = """
        SELECT stock_id,
               COUNT(*)                 AS total,
               COUNT(relevance)         AS with_relevance,
               COUNT(sentiment)         AS with_sentiment,
               MIN(date)                AS first_date,
               MAX(date)                AS last_date
        FROM news
        {where}
        GROUP BY stock_id
        ORDER BY stock_id
    """
    where, params = ("WHERE stock_id = %s", (stock_id,)) if stock_id else ("", ())
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql.format(where=where), params)
        return cur.fetchall()


# ---------- news：相關性（階段 ②） ----------


def fetch_news_without_relevance(stock_id=None, limit=None):
    """取出尚未判斷相關性的新聞（新到舊），附上股票名稱。"""
    sql = """
        SELECT n.id, n.stock_id, s.stock_name, n.title, n.content
        FROM news n JOIN stocks s ON s.stock_id = n.stock_id
        WHERE n.relevance IS NULL {and_stock}
        ORDER BY n.published_at DESC
        {limit}
    """
    params = []
    and_stock = ""
    if stock_id:
        and_stock = "AND n.stock_id = %s"
        params.append(stock_id)
    limit_sql = ""
    if limit:
        limit_sql = "LIMIT %s"
        params.append(int(limit))
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql.format(and_stock=and_stock, limit=limit_sql), params)
        return cur.fetchall()


def update_relevance(news_id, relevance, reason):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE news SET relevance = %s, relevance_reason = %s WHERE id = %s",
                (relevance, reason[:250], news_id),
            )
        conn.commit()


def relevance_summary(stock_id=None):
    """各股票的相關性分布。"""
    sql = """
        SELECT stock_id, relevance, COUNT(*) AS n
        FROM news {where}
        GROUP BY stock_id, relevance ORDER BY stock_id, relevance
    """
    where, params = ("WHERE stock_id = %s", (stock_id,)) if stock_id else ("", ())
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql.format(where=where), params)
        return cur.fetchall()


# ---------- news：情緒（階段 ③） ----------


def fetch_news_without_sentiment(stock_id=None, limit=None):
    """取出已判相關性（main / mention）但尚未計算情緒的新聞。"""
    sql = """
        SELECT id, stock_id, title, content
        FROM news
        WHERE sentiment IS NULL AND relevance IN ('main', 'mention') {and_stock}
        ORDER BY published_at DESC
        {limit}
    """
    params = []
    and_stock = ""
    if stock_id:
        and_stock = "AND stock_id = %s"
        params.append(stock_id)
    limit_sql = ""
    if limit:
        limit_sql = "LIMIT %s"
        params.append(int(limit))
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql.format(and_stock=and_stock, limit=limit_sql), params)
        return cur.fetchall()


def update_sentiments(rows):
    """
    批次寫入情緒結果。
    rows：list of (news_id, {"sentiment", "sentiment_label", "star_probs"})
    """
    sql = "UPDATE news SET sentiment = %s, sentiment_label = %s, star_probs = %s WHERE id = %s"
    params = [
        (r["sentiment"], r["sentiment_label"], json.dumps(r["star_probs"]), news_id)
        for news_id, r in rows
    ]
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, params)
        conn.commit()


def sentiment_summary(stock_id=None):
    """各股票、各相關性的情緒分布與平均分數。"""
    sql = """
        SELECT stock_id, relevance, sentiment_label, COUNT(*) AS n, AVG(sentiment) AS avg_s
        FROM news
        WHERE sentiment IS NOT NULL {and_stock}
        GROUP BY stock_id, relevance, sentiment_label
        ORDER BY stock_id, relevance, sentiment_label
    """
    and_stock, params = ("AND stock_id = %s", (stock_id,)) if stock_id else ("", ())
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql.format(and_stock=and_stock), params)
        return cur.fetchall()


# ---------- news：TAIDE 情緒（階段 3b，只處理 main） ----------


def fetch_news_without_llm_sentiment(stock_id=None, limit=None):
    """取出 relevance = main 且尚未由 TAIDE 判斷情緒的新聞，附上股票名稱。"""
    sql = """
        SELECT n.id, n.stock_id, s.stock_name, n.title, n.content
        FROM news n JOIN stocks s ON s.stock_id = n.stock_id
        WHERE n.relevance = 'main' AND n.llm_sentiment_label IS NULL {and_stock}
        ORDER BY n.published_at DESC
        {limit}
    """
    params = []
    and_stock = ""
    if stock_id:
        and_stock = "AND n.stock_id = %s"
        params.append(stock_id)
    limit_sql = ""
    if limit:
        limit_sql = "LIMIT %s"
        params.append(int(limit))
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql.format(and_stock=and_stock, limit=limit_sql), params)
        return cur.fetchall()


def update_llm_sentiment(news_id, label, reason):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE news SET llm_sentiment_label = %s, llm_sentiment_reason = %s WHERE id = %s",
                (label, reason[:250], news_id),
            )
        conn.commit()


def llm_vs_nlptown(stock_id=None):
    """TAIDE 與 nlptown 標籤的交叉表（只看兩者都有結果的 main 新聞）。"""
    sql = """
        SELECT llm_sentiment_label AS llm, sentiment_label AS nlptown, COUNT(*) AS n
        FROM news
        WHERE relevance = 'main' AND llm_sentiment_label IS NOT NULL
              AND sentiment_label IS NOT NULL {and_stock}
        GROUP BY llm, nlptown ORDER BY llm, nlptown
    """
    and_stock, params = ("AND stock_id = %s", (stock_id,)) if stock_id else ("", ())
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql.format(and_stock=and_stock), params)
        return cur.fetchall()


# ---------- news：向量化（階段 4） ----------


def fetch_news_for_index(stock_id=None):
    """取出要寫入向量庫的新聞（relevance 為 main / mention）。"""
    sql = """
        SELECT id, stock_id, date, source, title, url, content, relevance
        FROM news
        WHERE relevance IN ('main', 'mention') {and_stock}
        ORDER BY published_at DESC
    """
    and_stock, params = ("AND stock_id = %s", (stock_id,)) if stock_id else ("", ())
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql.format(and_stock=and_stock), params)
        return cur.fetchall()


# ---------- 換模型重新判斷（ingest.py relabel） ----------


def fetch_labels(stock_id):
    """某檔股票所有新聞目前的相關性與 LLM 情緒，重新判斷前備份、判斷後比較用。"""
    sql = """
        SELECT id, stock_id, date, title, relevance, relevance_reason,
               llm_sentiment_label, llm_sentiment_reason
        FROM news WHERE stock_id = %s ORDER BY published_at DESC
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (stock_id,))
        return cur.fetchall()


def reset_labels(stock_id):
    """清空某檔股票的相關性與 LLM 情緒，讓 relevance / llm-sentiment 階段重新判斷；回傳影響篇數。
    nlptown 的 sentiment 欄位不動（只是評估用的 baseline）。"""
    with get_connection() as conn:
        with conn.cursor() as cur:
            n = cur.execute(
                """UPDATE news SET relevance = NULL, relevance_reason = NULL,
                                   llm_sentiment_label = NULL, llm_sentiment_reason = NULL
                   WHERE stock_id = %s""",
                (stock_id,),
            )
        conn.commit()
    return n


def reset_llm_sentiment(stock_id):
    """只清空某檔股票的 LLM 情緒（相關性不動），換情緒 prompt 後重跑用；回傳影響篇數。"""
    with get_connection() as conn:
        with conn.cursor() as cur:
            n = cur.execute(
                "UPDATE news SET llm_sentiment_label = NULL, llm_sentiment_reason = NULL WHERE stock_id = %s",
                (stock_id,),
            )
        conn.commit()
    return n


# ---------- 趨勢與頁面 ----------


def fetch_main_sentiment(stock_id, start, end):
    """區間內 relevance = main 且已有 TAIDE 情緒的新聞（新到舊）。"""
    sql = """
        SELECT id, date, published_at, source, title, url,
               llm_sentiment_label, llm_sentiment_reason
        FROM news
        WHERE stock_id = %s AND relevance = 'main' AND llm_sentiment_label IS NOT NULL
              AND date BETWEEN %s AND %s
        ORDER BY published_at DESC
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (stock_id, start, end))
        return cur.fetchall()


def fetch_news_by_ids(ids):
    """依 id 取新聞的日期與 TAIDE 情緒（RAG 檢索後回查用）。"""
    if not ids:
        return []
    placeholders = ", ".join(["%s"] * len(ids))
    sql = f"""
        SELECT id, date, llm_sentiment_label, llm_sentiment_reason
        FROM news WHERE id IN ({placeholders})
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, list(ids))
        return cur.fetchall()


def fetch_news_contents(ids):
    """依 id 取新聞全文與目前的判斷結果（compare_models.py 比較模型與 prompt 用）"""
    if not ids:
        return []
    placeholders = ", ".join(["%s"] * len(ids))
    sql = f"""
        SELECT id, stock_id, date, title, content, relevance, llm_sentiment_label, llm_sentiment_reason
        FROM news WHERE id IN ({placeholders})
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, list(ids))
        return cur.fetchall()


def list_stocks_with_news():
    """有新聞的股票（頁面下拉選單用），依新聞篇數多到少。"""
    sql = """
        SELECT s.stock_id, s.stock_name, COUNT(*) AS n
        FROM news n JOIN stocks s ON s.stock_id = n.stock_id
        GROUP BY s.stock_id, s.stock_name
        ORDER BY n DESC
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchall()


def latest_news_date(stock_id):
    """某檔股票最新一篇新聞的日期；沒有新聞時回傳 None。"""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT MAX(date) AS d FROM news WHERE stock_id = %s", (stock_id,))
        row = cur.fetchone()
        return row["d"] if row else None
