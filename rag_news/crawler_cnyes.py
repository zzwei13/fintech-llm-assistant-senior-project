"""
鉅亨網（cnyes）新聞爬蟲

改寫自舊版 StockNewsCrawling/crawler_cnye_to_supa.py：
  - 搜尋頁需要捲動載入 → Selenium（無頭模式）
  - 文章頁內容在 HTML 中 → requests，速度快、較穩定
  - 不依賴自動產生的 class 名稱（如 jsx-1986041679），改用網址格式、<h1>、<time>、<article>
  - 一篇一筆（修正舊版同日內文累加）、保留標題與網址

單獨測試：
    python crawler_cnyes.py --stock 2330 --keyword 台積電 --days 1 --out sample_cnyes_2330.json
"""

import argparse
import json
import logging
import random
import re
import time
from datetime import date, datetime, timedelta
from urllib.parse import quote
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

from crawler_ltn import _is_near_duplicate, _shingles

log = logging.getLogger(__name__)

SEARCH_URL = "https://www.cnyes.com/search/news?keyword={keyword}"
ARTICLE_RE = re.compile(r"^https://news\.cnyes\.com/news/id/(\d+)")
TAIPEI = ZoneInfo("Asia/Taipei")
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}

# 內文清理規則
NAV_WORDS = {"上一篇", "下一篇", "延伸閱讀", "相關新聞", "更多新聞"}
# 署名行：「鉅亨網編譯某某 2026-10-26 21:40」「金色財經2026-10-25 17:30」等，短段落且以日期時間結尾
BYLINE_RE = re.compile(r"^.{0,30}\d{4}-\d{2}-\d{2} \d{2}:\d{2}$")
NUMBER_RE = re.compile(r"^[-+]?[\d,.]+%?$")  # 側欄股價、漲跌幅
SOURCE_NOTE_RE = re.compile(r"資料來源[:：]")  # 圖表資料來源
# 圖說：「……（圖：Shutterstock）」「……(圖：REUTERS/TPG)」「……(鉅亨網資料照)」
CAPTION_RE = re.compile(
    r"[（(](圖|圖片|照片)[:：][^）)]*[）)]$|[（(][^（）()]{0,10}資料照[）)]$"
)
MAX_CAPTION_LEN = 150
# 轉載文章與廣告的固定文字
BOILERPLATE = (
    "發佈者對本文章的內容承擔全部責任",
    "暢行幣圈交易全攻略",
    "LINE 社群",
    "LINE社群",
    "▌",
)
DISCLAIMER_PREFIXES = ("※免責聲明", "※ 免責聲明", "免責聲明：")
MIN_PARAGRAPH_LEN = 15
PAGE_LOAD_TIMEOUT = 20  # 秒

session = requests.Session()
session.headers.update(HEADERS)


def _article_id(url):
    m = ARTICLE_RE.match(url)
    return int(m.group(1)) if m else None


def _make_driver():
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    opt = Options()
    opt.add_argument("--headless=new")
    opt.add_argument("--disable-gpu")
    opt.add_argument(
        "--blink-settings=imagesEnabled=false"
    )  # 不載圖片，降低記憶體用量（減少分頁崩潰）
    opt.add_argument(f"user-agent={HEADERS['User-Agent']}")
    # 不等廣告、圖片等資源全部載完，HTML 解析完成就繼續（避免卡在載不完的第三方資源）
    opt.page_load_strategy = "eager"
    driver = webdriver.Chrome(options=opt)
    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
    return driver


def _collect_links(driver, found):
    """
    把目前頁面上的文章連結加入 found（id → 網址），回傳新增的數量。
    用一次 JavaScript 取出所有網址，不逐一讀取元素：捲動載入時頁面會重繪，
    逐一讀取容易遇到 StaleElementReferenceException（元素已被移除）。
    """
    new = 0
    hrefs = driver.execute_script(
        "return Array.from(document.querySelectorAll('a[href]'), a => a.href);"
    )
    for href in hrefs or []:
        href = (href or "").split("?")[0]
        aid = _article_id(href)
        if aid and aid not in found:
            found[aid] = href
            new += 1
    return new


def search_article_urls(keyword, start, max_scrolls=80, pause=1.5, check_every=5):
    """
    Selenium 捲動搜尋頁收集文章網址，回傳依 id 由新到舊排序的清單。
    每捲動 check_every 次，抓目前最舊那篇的發布日期；早於 start 就停止捲動。
    """
    driver = _make_driver()
    found = {}
    try:
        driver.get(SEARCH_URL.format(keyword=quote(keyword)))
        time.sleep(3)
        _collect_links(driver, found)
        no_new_rounds = 0

        for i in range(1, max_scrolls + 1):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(pause)
            if _collect_links(driver, found) == 0:
                no_new_rounds += 1
                if no_new_rounds >= 3:
                    log.info("連續 3 次捲動沒有新文章，停止")
                    break
            else:
                no_new_rounds = 0

            if i % check_every == 0 and found:
                oldest_date = _fetch_date(found[min(found)])
                if oldest_date and oldest_date < start:
                    log.info("已捲動到 %s 之前的文章，停止", start)
                    break
        else:
            log.warning("已達捲動上限 %d 次，結果可能不完整", max_scrolls)
    finally:
        driver.quit()

    urls = [found[k] for k in sorted(found, reverse=True)]
    log.info("關鍵字「%s」共找到 %d 篇鉅亨文章", keyword, len(urls))
    return urls


def _fetch_date(url):
    """只取文章的發布日期（台灣時間），供判斷是否停止捲動；失敗回傳 None。"""
    try:
        soup = BeautifulSoup(session.get(url, timeout=10).content, "html.parser")
        tag = soup.find("time")
        if tag and tag.get("datetime"):
            dt = datetime.fromisoformat(tag["datetime"].replace("Z", "+00:00"))
            return dt.astimezone(TAIPEI).date()
    except requests.RequestException:
        pass
    return None


def _clean_paragraphs(body):
    paras = []
    for p in body.find_all("p"):
        text = p.get_text(strip=True)
        if not text or text in NAV_WORDS:
            continue
        if len(text) < MIN_PARAGRAPH_LEN or NUMBER_RE.match(text):
            continue
        if len(text) <= 40 and BYLINE_RE.match(text):
            continue
        if SOURCE_NOTE_RE.search(text) and len(text) < 100:
            continue
        if len(text) <= MAX_CAPTION_LEN and CAPTION_RE.search(text):
            continue
        if any(k in text for k in BOILERPLATE) or text.startswith(DISCLAIMER_PREFIXES):
            continue
        paras.append(text)
    return paras


def _parse_html(html, url):
    """
    從 HTML 解析文章；回傳 (article, 原因)。
    article 為 None 時，原因說明缺了什麼，方便判斷是否要改用瀏覽器渲染。
    """
    soup = BeautifulSoup(html, "html.parser")

    time_tag = soup.find("time")
    if not (time_tag and time_tag.get("datetime")):
        return None, "沒有發布時間"
    # 例：2026-10-26T13:40:46.000Z（UTC）→ 台灣時間
    published_at = (
        datetime.fromisoformat(time_tag["datetime"].replace("Z", "+00:00"))
        .astimezone(TAIPEI)
        .replace(tzinfo=None)
    )

    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else ""
    if not title or title.strip(".…") == "":  # 「...」是尚未載入完成的佔位符
        og = soup.find("meta", property="og:title")
        title = (og.get("content") or "").split(" | ")[0].strip() if og else ""
    if not title:
        return None, "沒有標題"

    body = soup.find("article")
    content = "\n".join(_clean_paragraphs(body)) if body else ""
    if not content:
        return None, "沒有內文（可能需要 JavaScript 渲染）"

    return {
        "source": "cnyes",
        "url": url,
        "title": title,
        "published_at": published_at,
        "date": published_at.date(),
        "content": content,
    }, ""


RESTART_EVERY = 50  # 每渲染幾篇就重開瀏覽器，避免長時間執行記憶體累積導致崩潰


class DriverBox:
    """
    Selenium 瀏覽器的容器：用到時才開啟；瀏覽器崩潰（invalid session id、disconnected）時自動重開，
    並且每渲染 RESTART_EVERY 篇主動重開一次。
    否則瀏覽器一旦崩潰，後面所有需要渲染的文章都會失敗。
    """

    def __init__(self):
        self.driver = None
        self.renders = 0

    def get(self):
        if self.driver is None or self.renders >= RESTART_EVERY:
            self.restart()
        self.renders += 1
        return self.driver

    def restart(self):
        self.quit()
        self.driver = _make_driver()
        self.renders = 0

    def quit(self):
        if self.driver is not None:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None


def _is_dead_session(e):
    msg = str(e).lower()
    return (
        "invalid session id" in msg or "disconnected" in msg or "no such window" in msg
    )


def _render(url, box, render_wait):
    from selenium.common.exceptions import TimeoutException

    driver = box.get()
    try:
        driver.get(url)
    except TimeoutException:
        log.debug("頁面載入逾時，改用已載入的內容：%s", url)  # 正文通常已經出現
    deadline = time.time() + render_wait
    while time.time() < deadline:  # 等到正文段落出現
        if driver.find_elements("css selector", "article p"):
            break
        time.sleep(0.5)
    time.sleep(0.5)
    return _parse_html(driver.page_source, url)


def parse_article(url, driver=None, render_wait=8):
    """
    解析單篇文章：先用 requests；內容不完整且有提供 driver（DriverBox）時，改用 Selenium 渲染後再解析。
    瀏覽器崩潰時重開一次再試。欄位仍不完整時回傳 None。
    """
    try:
        resp = session.get(url, timeout=10)
        resp.raise_for_status()
        article, reason = _parse_html(resp.content, url)
    except requests.RequestException as e:
        article, reason = None, f"請求失敗：{e}"

    if article is None and driver is not None:
        for attempt in range(2):
            try:
                article, reason = _render(url, driver, render_wait)
                if article:
                    log.debug("改用瀏覽器渲染成功：%s", url)
                break
            except Exception as e:
                reason = f"瀏覽器渲染失敗：{str(e).splitlines()[0][:120]}"  # 只留第一行，不印整段 stacktrace
                if attempt == 0 and _is_dead_session(e):
                    log.info("瀏覽器崩潰，重新開啟後重試：%s", url)
                    driver.restart()
                    continue
                break

    if article is None:
        log.info("略過（%s）：%s", reason, url)
    return article


def crawl(
    stock_id,
    keyword,
    start,
    end,
    min_length=100,
    skip_urls=None,
    delay=(0.8, 1.5),
    stop_after_old=5,
):
    """
    抓取某檔股票在日期區間內的鉅亨新聞，逐篇 yield 一筆 dict（格式與 crawler_ltn 相同）。
    文章依新到舊處理，連續 stop_after_old 篇早於 start 就停止。
    """
    skip_urls = skip_urls or set()
    seen_shingles = []
    old_streak = 0
    urls = search_article_urls(keyword, start)
    box = (
        DriverBox()
    )  # 內文需要 JavaScript 渲染時的備援；用到時才開瀏覽器，崩潰時自動重開

    try:
        yield from _crawl_urls(
            urls,
            stock_id,
            start,
            end,
            min_length,
            skip_urls,
            delay,
            stop_after_old,
            seen_shingles,
            box,
        )
    finally:
        box.quit()


def _crawl_urls(
    urls,
    stock_id,
    start,
    end,
    min_length,
    skip_urls,
    delay,
    stop_after_old,
    seen_shingles,
    driver,
):
    old_streak = 0
    for url in urls:
        if url in skip_urls:
            continue
        article = parse_article(url, driver=driver)
        time.sleep(random.uniform(*delay))
        if article is None:
            continue

        if article["date"] < start:
            old_streak += 1
            if old_streak >= stop_after_old:
                log.info("連續 %d 篇早於 %s，停止", stop_after_old, start)
                break
            continue
        old_streak = 0
        if article["date"] > end:
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
    parser = argparse.ArgumentParser(description="鉅亨網新聞爬蟲（測試用）")
    parser.add_argument("--stock", required=True)
    parser.add_argument("--keyword", required=True)
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--out")
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
