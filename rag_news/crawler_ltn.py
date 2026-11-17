"""
自由時報（LTN）財經新聞爬蟲

依「關鍵字＋日期區間」搜尋，進入每篇自由財經（ec.ltn.com.tw）文章抓取
標題、發布時間與內文。一篇新聞回傳一筆資料，不做跨文章累加。

單獨測試：
    python crawler_ltn.py --stock 2330 --keyword 台積電 --days 7 --out sample_2330.json
"""

import argparse
import json
import logging
import random
import re
import time
from datetime import date, datetime, timedelta

import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

SEARCH_URL = "https://search.ltn.com.tw/list"
ARTICLE_PREFIX = "https://ec.ltn.com.tw/article/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}

TIME_RE = re.compile(r"\d{4}/\d{2}/\d{2} \d{2}:\d{2}")
LEAD_RE = re.compile(r"^〔[^〕]*〕")  # 例：〔財經頻道／綜合報導〕
BOILERPLATE = (
    "請繼續往下閱讀",
    "點我訂閱",
    "點我下載APP",
    "按我看活動辦法",
    "一手掌握經濟脈動",
)
# 圖說：短段落、以圖片來源括號結尾，例：「……（資料照）」「……（美聯社）」「……（記者某某攝）」
CAPTION_RE = re.compile(r"（[^（）]{0,20}(資料照|攝|翻攝|社|路透|彭博|取自[^（）]*)）$")
# 署名行：「記者某某／某地報導」「■某某」
BYLINE_RE = re.compile(r"^(記者\S{1,10}／\S{0,15}報導|■\S{1,10})$")
# 文末作者介紹：「（作者為……）」
AUTHOR_NOTE_RE = re.compile(r"^（作者為.*）$")
MAX_CAPTION_LEN = 150

session = requests.Session()
session.headers.update(HEADERS)


def _get(url, params=None, retries=3):
    """GET 請求，失敗時指數退避重試；404 代表頁面不存在，直接回傳 None 不重試。"""
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=10)
            if resp.status_code == 404:
                log.debug("頁面不存在（404）：%s", resp.url)
                return None
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            wait = 2**attempt
            log.warning("請求失敗（%s），%d 秒後重試：%s", e, wait, url)
            time.sleep(wait)
    log.error("放棄請求：%s", url)
    return None


def search_article_urls(keyword, start, end, max_pages=200, delay=(0.8, 1.5)):
    """搜尋頁依日期區間翻頁，回傳去重後的自由財經文章網址（依搜尋結果順序）。"""
    urls, seen = [], set()
    prev_page_links = None

    for page in range(1, max_pages + 1):
        resp = _get(
            SEARCH_URL,
            params={
                "keyword": keyword,
                "start_time": start.strftime("%Y%m%d"),
                "end_time": end.strftime("%Y%m%d"),
                "sort": "date",
                "type": "all",
                "page": page,
            },
        )
        if resp is None:  # 404（超過最後一頁）或連線失敗
            break

        soup = BeautifulSoup(resp.content, "html.parser")
        page_links = [a["href"] for a in soup.select("div.cont a[href]")]

        # 沒有結果，或翻到與上一頁相同的內容，代表已經是最後一頁
        if not page_links or page_links == prev_page_links:
            break
        prev_page_links = page_links

        for href in page_links:
            if href.startswith(ARTICLE_PREFIX) and href not in seen:
                seen.add(href)
                urls.append(href)

        time.sleep(random.uniform(*delay))
    else:
        log.warning(
            "關鍵字「%s」已達翻頁上限 %d 頁，結果可能不完整", keyword, max_pages
        )

    log.info("關鍵字「%s」共找到 %d 篇自由財經文章", keyword, len(urls))
    return urls


def _clean_paragraphs(body):
    """取出正文段落：跳過廣告、圖說與固定宣傳文字，去掉導言前綴。"""
    paras = []
    for p in body.find_all("p"):
        if p.get("class"):  # 例：before_ir、appE1121 等廣告區塊
            continue
        text = p.get_text(strip=True)
        if not text or any(k in text for k in BOILERPLATE):
            continue
        if BYLINE_RE.match(text) or AUTHOR_NOTE_RE.match(text):
            continue
        if len(text) <= MAX_CAPTION_LEN and CAPTION_RE.search(text):
            continue
        paras.append(text)

    # 圖說通常在導言（〔…報導〕）之前，從導言開始取
    lead_idx = next((i for i, t in enumerate(paras) if LEAD_RE.match(t)), None)
    if lead_idx is not None:
        paras = paras[lead_idx:]
        paras[0] = LEAD_RE.sub("", paras[0]).strip()

    return [t for t in paras if t]


def parse_article(url):
    """解析單篇文章；欄位不完整時回傳 None。"""
    resp = _get(url)
    if resp is None:
        return None
    soup = BeautifulSoup(resp.content, "html.parser")

    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else ""

    published_at = None
    for span in soup.select("span.time"):
        m = TIME_RE.search(span.get_text(strip=True))
        if m:
            published_at = datetime.strptime(m.group(), "%Y/%m/%d %H:%M")
            break

    bodies = soup.select("div.text")
    if not bodies:
        return None
    body = max(bodies, key=lambda d: len(d.find_all("p")))
    content = "\n".join(_clean_paragraphs(body))

    if not (title and published_at and content):
        log.warning("欄位不完整，略過：%s", url)
        return None

    return {
        "source": "ltn",
        "url": url,
        "title": title,
        "published_at": published_at,
        "date": published_at.date(),
        "content": content,
    }


def _shingles(text, n=5, length=400):
    """取內文開頭的字元 n-gram 集合，用來比對兩篇是否為同一篇文章。"""
    text = re.sub(r"\s+", "", text[:length])
    return {text[i : i + n] for i in range(max(len(text) - n + 1, 1))}


def _is_near_duplicate(shingles, seen, threshold=0.6):
    """與已收錄文章的 Jaccard 相似度超過門檻，視為重複（例：同一篇的電子版與報紙版）。"""
    return any(len(shingles & s) / len(shingles | s) >= threshold for s in seen)


def crawl(
    stock_id, keyword, start, end, min_length=100, skip_urls=None, delay=(0.8, 1.5)
):
    """
    抓取某檔股票在日期區間內的新聞，逐篇 yield 一筆 dict。
    skip_urls：已存在資料庫的網址，直接略過不抓內文。
    delay：每次請求之間隨機等待的秒數範圍，量大時可拉長以降低被擋的機率。
    """
    skip_urls = skip_urls or set()
    seen_shingles = []
    for url in search_article_urls(keyword, start, end, delay=delay):
        if url in skip_urls:
            continue
        article = parse_article(url)
        time.sleep(random.uniform(*delay))
        if article is None:
            continue
        if not (start <= article["date"] <= end):
            continue
        if len(article["content"]) < min_length:
            log.info("內文過短，略過：%s", url)
            continue
        shingles = _shingles(article["content"])
        if _is_near_duplicate(shingles, seen_shingles):
            log.info("內容與已收錄文章重複，略過：%s", url)
            continue
        seen_shingles.append(shingles)
        article["stock_id"] = stock_id
        yield article


def main():
    parser = argparse.ArgumentParser(description="LTN 財經新聞爬蟲（測試用）")
    parser.add_argument("--stock", required=True, help="股票代號，例：2330")
    parser.add_argument("--keyword", required=True, help="搜尋關鍵字，例：台積電")
    parser.add_argument("--days", type=int, default=7, help="往回抓幾天")
    parser.add_argument("--out", help="輸出 JSON 檔（不指定則只印摘要）")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    end = date.today()
    start = end - timedelta(days=args.days)

    articles = list(crawl(args.stock, args.keyword, start, end))
    for a in articles:
        print(f"{a['date']} | {a['title'][:40]} | {len(a['content'])} 字")
    print(f"\n共 {len(articles)} 篇（{start} ~ {end}）")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2, default=str)
        print(f"已存到 {args.out}")


if __name__ == "__main__":
    main()
