"""
新聞資料管線（ingest），分階段執行，每個階段可單獨重跑：

    ① crawl      爬蟲 → 寫入 MySQL news（基本欄位）
    ② relevance  LLM（.env 的 OLLAMA_MODEL）判斷相關性
    ③ sentiment      nlptown 計算情緒（main / mention），baseline
    ③b llm-sentiment LLM 判斷對該公司的利多／利空／中立（只限 main），主要指標
    ④ index      bge-m3 向量化 → Chroma（main / mention）
    relabel      換模型後全部重新判斷：備份 → 清空 → ②、③b 重跑 → 同步 Chroma → 比較新舊結果

用法：
    python ingest.py crawl --stock 2330 --days 30
    python ingest.py crawl --stock 2330 2002 3443 --days 30
    python ingest.py crawl --stock-file stocks_30.txt --days 30 --delay 2 4
    python ingest.py crawl --stock 2330 --days 1 --source cnyes      # 只跑鉅亨
    python ingest.py relevance --stock 2330 --limit 20             # 先小量測試
    python ingest.py relevance                                     # 全部未判斷的新聞
    python ingest.py sentiment --stock 2330 --limit 20
    python ingest.py sentiment
    python ingest.py index
    python ingest.py status
    python ingest.py relabel --stock 2330 2317 2002            # 換模型後全部重跑（會先備份）
    python ingest.py relabel --stock 2330 2317 2002 --resume   # 中斷後接著跑，不再清空
    python ingest.py relabel --stock 2330 2317 --sentiment-only  # 只換情緒 prompt：只重跑情緒
"""

import argparse
import csv
import glob
import os
import logging
import time
from datetime import date, datetime, timedelta

import db

SOURCES = ("ltn", "cnyes")


def get_crawler(source):
    """依來源取得爬蟲函式；鉅亨需要 Selenium，用到時才匯入。"""
    if source == "ltn":
        from crawler_ltn import crawl
    elif source == "cnyes":
        from crawler_cnyes import crawl
    else:
        raise ValueError(f"未知的新聞來源：{source}")
    return crawl


log = logging.getLogger("ingest")


def run_crawl(stock_ids, days, sources=SOURCES, delay=(0.8, 1.5), batch_size=10):
    end = date.today()
    start = end - timedelta(days=days)

    for i, stock_id in enumerate(stock_ids, 1):
        log.info("===== 第 %d/%d 檔 =====", i, len(stock_ids))
        stock_name = db.get_stock_name(stock_id)
        if stock_name is None:
            log.error("stocks 表中找不到代號 %s，略過", stock_id)
            continue

        for source in sources:
            crawl = get_crawler(source)
            skip_urls = db.get_existing_urls(stock_id)
            log.info(
                "【%s %s｜%s】%s ~ %s，資料庫已有 %d 篇",
                stock_id,
                stock_name,
                source,
                start,
                end,
                len(skip_urls),
            )

            batch, inserted = [], 0
            try:
                for article in crawl(
                    stock_id, stock_name, start, end, skip_urls=skip_urls, delay=delay
                ):
                    batch.append(article)
                    if len(batch) >= batch_size:  # 分批寫入，中途中斷也不會全部白跑
                        inserted += db.insert_news(batch)
                        batch = []
            except Exception:
                log.exception(
                    "【%s %s｜%s】爬蟲發生錯誤，已抓到的部分仍會寫入",
                    stock_id,
                    stock_name,
                    source,
                )
            inserted += db.insert_news(batch)

            log.info("【%s %s｜%s】新增 %d 篇", stock_id, stock_name, source, inserted)


def run_relevance(stock_id=None, limit=None):
    from relevance import MODEL, classify

    rows = db.fetch_news_without_relevance(stock_id, limit)
    log.info("待判斷 %d 篇（模型：%s）", len(rows), MODEL)
    started, done, failed, by_rule = time.time(), 0, 0, 0

    for i, r in enumerate(rows, 1):
        try:
            result = classify(r["stock_id"], r["stock_name"], r["title"], r["content"])
        except Exception as e:
            failed += 1
            log.warning(
                "判斷失敗（%s），略過 id=%s：%s",
                type(e).__name__,
                r["id"],
                r["title"][:30],
            )
            continue
        db.update_relevance(
            r["id"], result.relevance, result.reason
        )  # 逐篇寫入，可隨時中斷續跑
        done += 1
        by_rule += result.by_rule
        src = "規則" if result.by_rule else "模型"
        log.info(
            "[%d/%d] %-9s（%s）%s %s",
            i,
            len(rows),
            result.relevance,
            src,
            r["stock_id"],
            r["title"][:30],
        )

    elapsed = time.time() - started
    if done:
        log.info(
            "完成 %d 篇（規則 %d 篇、模型 %d 篇）、失敗 %d 篇，平均每篇 %.1f 秒",
            done,
            by_rule,
            done - by_rule,
            failed,
            elapsed / done,
        )

    print("\n相關性分布：")
    for s in db.relevance_summary(stock_id):
        print(f"  {s['stock_id']}  {str(s['relevance']):<10}{s['n']:>5}")


def run_sentiment(stock_id=None, limit=None, batch_size=16):
    from sentiment import MODEL, analyze, build_text

    rows = db.fetch_news_without_sentiment(stock_id, limit)
    log.info("待計算情緒 %d 篇（模型：%s）", len(rows), MODEL)
    started, done = time.time(), 0

    for i in range(0, len(rows), batch_size):  # 每批寫入一次，中斷後可續跑
        batch = rows[i : i + batch_size]
        results = analyze(
            [build_text(r["title"], r["content"]) for r in batch], batch_size=batch_size
        )
        db.update_sentiments([(r["id"], res) for r, res in zip(batch, results)])
        done += len(batch)
        log.info("[%d/%d] 已完成", done, len(rows))

    if done:
        log.info("完成 %d 篇，平均每篇 %.2f 秒", done, (time.time() - started) / done)

    print("\n情緒分布：")
    print(f"  {'股票':<6}{'相關性':<10}{'標籤':<10}{'篇數':>5}{'平均分數':>9}")
    for s in db.sentiment_summary(stock_id):
        print(
            f"  {s['stock_id']:<6}{s['relevance']:<10}{s['sentiment_label']:<10}{s['n']:>5}{s['avg_s']:>+9.2f}"
        )


def run_llm_sentiment(stock_id=None, limit=None):
    from llm_sentiment import MODEL, classify

    rows = db.fetch_news_without_llm_sentiment(stock_id, limit)
    log.info("待判斷情緒（LLM，只限 main）%d 篇（模型：%s）", len(rows), MODEL)
    started, done, failed = time.time(), 0, 0

    for i, r in enumerate(rows, 1):
        try:
            result = classify(r["stock_id"], r["stock_name"], r["title"], r["content"])
        except Exception as e:
            failed += 1
            log.warning(
                "判斷失敗（%s），略過 id=%s：%s",
                type(e).__name__,
                r["id"],
                r["title"][:30],
            )
            continue
        db.update_llm_sentiment(
            r["id"], result.label, result.reason
        )  # 逐篇寫入，可隨時中斷續跑
        done += 1
        log.info(
            "[%d/%d] %-9s %s %s",
            i,
            len(rows),
            result.label,
            r["stock_id"],
            r["title"][:30],
        )

    if done:
        log.info(
            "完成 %d 篇、失敗 %d 篇，平均每篇 %.1f 秒",
            done,
            failed,
            (time.time() - started) / done,
        )

    print("\nLLM（列）vs nlptown（欄）：")
    for s in db.llm_vs_nlptown(stock_id):
        print(f"  {s['llm']:<9} {s['nlptown']:<9}{s['n']:>5}")


def run_index(stock_id=None):
    from vectorstore import EMBEDDING_MODEL, PERSIST_DIR, add_news, existing_ids

    rows = db.fetch_news_for_index(stock_id)
    log.info(
        "待索引新聞 %d 篇（模型：%s，向量庫：%s）",
        len(rows),
        EMBEDDING_MODEL,
        PERSIST_DIR,
    )
    started = time.time()
    n_news, n_chunks = add_news(rows)
    log.info(
        "新增 %d 篇、%d 個片段，花費 %.0f 秒；向量庫共 %d 個片段",
        n_news,
        n_chunks,
        time.time() - started,
        len(existing_ids()),
    )


BACKUP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backup")
LABEL_FIELDS = [
    "id",
    "stock_id",
    "date",
    "title",
    "relevance",
    "relevance_reason",
    "llm_sentiment_label",
    "llm_sentiment_reason",
]


def backup_labels(stock_ids):
    """把目前的相關性與情緒存成 backup/labels_<時間>.csv，回傳路徑"""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    path = os.path.join(BACKUP_DIR, f"labels_{datetime.now():%Y%m%d_%H%M}.csv")
    with open(
        path, "w", newline="", encoding="utf-8-sig"
    ) as f:  # utf-8-sig：Excel 開啟不亂碼
        w = csv.DictWriter(f, LABEL_FIELDS)
        w.writeheader()
        for sid in stock_ids:
            w.writerows(db.fetch_labels(sid))
    return path


def latest_backup():
    files = sorted(glob.glob(os.path.join(BACKUP_DIR, "labels_*.csv")))
    return files[-1] if files else None


def compare_labels(backup_path, stock_ids):
    """
    比較備份（舊模型）和目前（新模型）的結果：
      相關性：舊 → 新 的篇數
      情緒：新舊都是 main 的新聞中，情緒改變的篇數
    明細寫到 backup/changes_<時間>.csv
    """
    with open(backup_path, encoding="utf-8-sig") as f:
        old = {int(r["id"]): r for r in csv.DictReader(f)}

    changes = []
    for sid in stock_ids:
        rel_moves, sent_moves, both_main = {}, {}, 0
        for r in db.fetch_labels(sid):
            o = old.get(r["id"])
            if o is None:
                continue
            o_rel, n_rel = o["relevance"] or "—", r["relevance"] or "—"
            rel_moves[(o_rel, n_rel)] = rel_moves.get((o_rel, n_rel), 0) + 1
            o_sent, n_sent = (
                o["llm_sentiment_label"] or "—",
                r["llm_sentiment_label"] or "—",
            )
            if o_rel == n_rel == "main":
                both_main += 1
                if o_sent != n_sent:
                    sent_moves[(o_sent, n_sent)] = (
                        sent_moves.get((o_sent, n_sent), 0) + 1
                    )
            if o_rel != n_rel or o_sent != n_sent:
                changes.append(
                    {
                        "id": r["id"],
                        "stock_id": sid,
                        "date": r["date"],
                        "title": r["title"],
                        "舊相關性": o_rel,
                        "新相關性": n_rel,
                        "舊情緒": o_sent,
                        "新情緒": n_sent,
                        "舊理由": o["llm_sentiment_reason"],
                        "新理由": r["llm_sentiment_reason"],
                    }
                )

        print(f"\n{sid} 相關性（舊 → 新）：")
        for (a, b), n in sorted(rel_moves.items()):
            print(f"  {a:<9} → {b:<9}{n:>5}{'' if a == b else '  *'}")
        n_sent = sum(sent_moves.values())
        print(f"{sid} 情緒：新舊都是 main 的 {both_main} 篇中，改判 {n_sent} 篇")
        for (a, b), n in sorted(sent_moves.items(), key=lambda x: -x[1]):
            print(f"  {a:<9} → {b:<9}{n:>5}")

    path = os.path.join(BACKUP_DIR, f"changes_{datetime.now():%Y%m%d_%H%M}.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(
            f,
            [
                "id",
                "stock_id",
                "date",
                "title",
                "舊相關性",
                "新相關性",
                "舊情緒",
                "新情緒",
                "舊理由",
                "新理由",
            ],
        )
        w.writeheader()
        w.writerows(changes)
    print(f"\n改變的新聞明細：{path}")


def sync_index(stock_id):
    """相關性改變後同步向量庫：刪除變成 unrelated 的片段、更新 main/mention、補上新加入的新聞"""
    from vectorstore import add_news, sync_relevance

    relevance_by_id = {r["id"]: r["relevance"] for r in db.fetch_labels(stock_id)}
    deleted, updated = sync_relevance(stock_id, relevance_by_id)
    n_news, n_chunks = add_news(db.fetch_news_for_index(stock_id))
    log.info(
        "%s 向量庫：刪除 %d 個片段、更新 %d 個片段的相關性、新增 %d 篇（%d 個片段）",
        stock_id,
        deleted,
        updated,
        n_news,
        n_chunks,
    )


def run_relabel(stock_ids, resume=False, yes=False, sentiment_only=False):
    """
    換模型後全部重新判斷。逐篇寫入，中斷後用 --resume 接著跑（不會再清空）。
    sentiment_only：只清空、重跑情緒（換情緒 prompt 時用），相關性與向量庫不動
    """
    from llm_sentiment import PROMPT_VERSION
    from relevance import MODEL

    if resume:
        backup = latest_backup()
        log.info("接著上次的進度重跑（模型：%s），比較基準：%s", MODEL, backup)
    else:
        what = "情緒" if sentiment_only else "相關性與情緒"
        print(
            f"即將用 {MODEL}（情緒 prompt {PROMPT_VERSION}）重新判斷 {', '.join(stock_ids)} 的{what}，"
            "目前的結果會先備份到 backup/ 再清空。"
        )
        if not yes and input("輸入 y 繼續：").strip().lower() != "y":
            print("已取消")
            return
        backup = backup_labels(stock_ids)
        log.info("已備份到 %s", backup)
        for sid in stock_ids:
            if sentiment_only:
                log.info("%s 清空 %d 篇的情緒", sid, db.reset_llm_sentiment(sid))
            else:
                log.info("%s 清空 %d 篇的相關性與情緒", sid, db.reset_labels(sid))

    started = time.time()
    if not sentiment_only:
        for sid in stock_ids:
            run_relevance(sid)
    for sid in stock_ids:
        run_llm_sentiment(sid)
    if not sentiment_only:
        for sid in stock_ids:
            sync_index(sid)
    log.info("重新判斷完成，共花費 %.1f 分鐘", (time.time() - started) / 60)

    if backup:
        compare_labels(backup, stock_ids)
    print("\n下一步：python chain.py --stock <代號> --refresh 重新產生報導焦點")


def read_stock_file(path):
    """讀取股票代號清單：一行一個代號，# 開頭為註解，可在代號後加空白與名稱備註。"""
    with open(path, encoding="utf-8") as f:
        return [
            line.split()[0]
            for line in f
            if line.strip() and not line.lstrip().startswith("#")
        ]


def show_status():
    rows = db.count_news()
    if not rows:
        print("news 表目前沒有資料")
        return
    print(f"{'股票':<6}{'篇數':>6}{'已判相關性':>10}{'已算情緒':>8}  日期範圍")
    for r in rows:
        print(
            f"{r['stock_id']:<6}{r['total']:>6}{r['with_relevance']:>10}"
            f"{r['with_sentiment']:>8}  {r['first_date']} ~ {r['last_date']}"
        )


def main():
    parser = argparse.ArgumentParser(description="新聞資料管線")
    sub = parser.add_subparsers(dest="stage", required=True)

    p_crawl = sub.add_parser("crawl", help="爬蟲並寫入 MySQL")
    target = p_crawl.add_mutually_exclusive_group(required=True)
    target.add_argument("--stock", nargs="+", help="股票代號，可多個")
    target.add_argument("--stock-file", help="股票代號清單檔，一行一個")
    p_crawl.add_argument("--days", type=int, default=30, help="往回抓幾天")
    p_crawl.add_argument(
        "--source",
        nargs="+",
        choices=SOURCES,
        default=list(SOURCES),
        help="新聞來源，預設全部（ltn 自由時報、cnyes 鉅亨）",
    )
    p_crawl.add_argument(
        "--delay",
        type=float,
        nargs=2,
        default=[0.8, 1.5],
        metavar=("MIN", "MAX"),
        help="每次請求間隔的秒數範圍，量大時建議拉長，例如 --delay 2 4",
    )

    p_rel = sub.add_parser("relevance", help="LLM 判斷相關性")
    p_rel.add_argument("--stock", help="只處理某檔股票")
    p_rel.add_argument("--limit", type=int, help="最多處理幾篇（測試用）")

    p_sent = sub.add_parser("sentiment", help="nlptown 計算情緒")
    p_sent.add_argument("--stock", help="只處理某檔股票")
    p_sent.add_argument("--limit", type=int, help="最多處理幾篇（測試用）")

    p_llm = sub.add_parser("llm-sentiment", help="LLM 判斷情緒（只限 main）")
    p_llm.add_argument("--stock", help="只處理某檔股票")
    p_llm.add_argument("--limit", type=int, help="最多處理幾篇（測試用）")

    p_idx = sub.add_parser("index", help="bge-m3 向量化並寫入 Chroma")
    p_idx.add_argument("--stock", help="只處理某檔股票")

    p_relabel = sub.add_parser(
        "relabel", help="換模型後全部重新判斷相關性與情緒（會先備份）"
    )
    p_relabel.add_argument("--stock", nargs="+", required=True, help="股票代號，可多個")
    p_relabel.add_argument(
        "--resume", action="store_true", help="中斷後接著跑，不備份、不清空"
    )
    p_relabel.add_argument("--yes", action="store_true", help="不詢問確認")
    p_relabel.add_argument(
        "--sentiment-only", action="store_true", help="只重跑情緒（換情緒 prompt 時用）"
    )

    sub.add_parser("status", help="查看各股票的資料完成度")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.stage == "crawl":
        stock_ids = args.stock or read_stock_file(args.stock_file)
        run_crawl(stock_ids, args.days, sources=args.source, delay=tuple(args.delay))
        show_status()
    elif args.stage == "relevance":
        run_relevance(args.stock, args.limit)
        show_status()
    elif args.stage == "sentiment":
        run_sentiment(args.stock, args.limit)
        show_status()
    elif args.stage == "llm-sentiment":
        run_llm_sentiment(args.stock, args.limit)
        show_status()
    elif args.stage == "index":
        run_index(args.stock)
    elif args.stage == "relabel":
        run_relabel(args.stock, args.resume, args.yes, args.sentiment_only)
        show_status()
    elif args.stage == "status":
        show_status()


if __name__ == "__main__":
    main()
