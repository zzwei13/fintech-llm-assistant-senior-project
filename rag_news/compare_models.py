"""
比較模型與 prompt 版本：相關性判斷、目標導向情緒判斷

組合：每個模型 × 每個 prompt 版本（預設 TAIDE、Gemma 4 × v1、v2）
資料：
  1. 行為測試句（test_cases.py，答案已知）→ 正確率
  2. 真實新聞：chain.py 快取中報導焦點引用的新聞（預設台積電 2026-10-26）
     → 和資料庫目前的結果（TAIDE＋v1）比較，列出改判的新聞，讓人工檢查

結果只寫成 CSV，不寫進資料庫，不會覆蓋現有的判斷。
  compare/behavioral_<時間>.csv   行為測試：每題一列，各組合的答案與理由並排
  compare/real_<時間>.csv         真實新聞：每篇一列，資料庫目前的結果與各組合並排

執行（Ollama 要開著；4 個組合約 20～40 分鐘）：
    python compare_models.py
    python compare_models.py --models taide-8b:latest gemma4:e4b --versions v1 v2
    python compare_models.py --no-real          # 只跑行為測試，快速確認
    python compare_models.py --limit 10         # 真實新聞只取前 10 篇
    python compare_models.py --models gemma4:12b --versions v1 v3 --no-real   # 比較情緒 prompt v1、v3
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime

import db
import llm_sentiment
import relevance
from datetime import date

from chain import cache_path
from test_cases import RELEVANCE_CASES, SENTIMENT_CASES, STOCK_ID, STOCK_NAME

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare")


def timed(fn):
    """執行 fn()，回傳 (結果, 秒數, 錯誤訊息)"""
    t0 = time.time()
    try:
        return fn(), time.time() - t0, None
    except Exception as e:  # JSON 格式錯誤、連線失敗等
        return None, time.time() - t0, f"{type(e).__name__}: {e}"[:200]


def run_sentiment(
    model, version, title, content, stock_id=STOCK_ID, stock_name=STOCK_NAME
):
    out, sec, err = timed(
        lambda: llm_sentiment.classify(
            stock_id, stock_name, title, content, model=model, version=version
        )
    )
    return {
        "label": out.label if out else "ERROR",
        "subject": getattr(out, "subject", "") if out else "",
        "reason": out.reason if out else err,
        "sec": sec,
        "error": err is not None,
    }


def run_relevance(
    model, version, title, content, stock_id=STOCK_ID, stock_name=STOCK_NAME
):
    version = (
        version if version in relevance.PROMPTS else "v1"
    )  # 相關性沒有的版本（如 v3）用 v1
    out, sec, err = timed(
        lambda: relevance.classify(
            stock_id, stock_name, title, content, model=model, version=version
        )
    )
    return {
        "label": out.relevance if out else "ERROR",
        "reason": out.reason if out else err,
        "by_rule": bool(out and out.by_rule),
        "sec": sec,
        "error": err is not None,
    }


def load_real_news(stock_id, end, limit):
    """chain.py 快取中引用的新聞（報導焦點的成員），回傳資料庫中的全文與目前的判斷"""
    path = cache_path(stock_id, date.fromisoformat(end))
    if not os.path.exists(path):
        raise SystemExit(
            f"找不到 {path}，請先執行 python chain.py --stock {stock_id} --end {end}"
        )
    with open(path, encoding="utf-8") as f:
        ids = [s["news_id"] for s in json.load(f)["sources"]]
    rows = {r["id"]: r for r in db.fetch_news_contents(ids)}
    news = [rows[i] for i in ids if i in rows]
    return news[:limit] if limit else news


def combo_name(model, version):
    return f"{model.split(':')[0]}_{version}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="比較模型與 prompt 版本")
    parser.add_argument("--models", nargs="+", default=[relevance.MODEL, "gemma4:e4b"])
    parser.add_argument("--versions", nargs="+", default=["v1", "v2"])
    parser.add_argument("--stock", default=STOCK_ID)
    parser.add_argument("--end", default="2026-10-26", help="chain.py 快取的結束日期")
    parser.add_argument(
        "--no-real", action="store_true", help="不跑真實新聞，只跑行為測試"
    )
    parser.add_argument("--limit", type=int, help="真實新聞最多幾篇")
    args = parser.parse_args()

    stock_name = db.get_stock_name(args.stock)
    real = [] if args.no_real else load_real_news(args.stock, args.end, args.limit)
    combos = [
        (m, v) for m in args.models for v in args.versions
    ]  # 同一個模型連續跑，減少 Ollama 重新載入
    print(f"組合：{[combo_name(m, v) for m, v in combos]}")
    print(
        f"行為測試：情緒 {len(SENTIMENT_CASES)} 題、相關性 {len(RELEVANCE_CASES)} 題；真實新聞 {len(real)} 篇\n"
    )

    behavioral = {}  # (task, i) -> {combo: result}
    real_res = {}  # news_id -> {combo: {"rel":..., "sent":...}}
    summary = []

    for model, version in combos:
        name = combo_name(model, version)
        t0 = time.time()
        secs, errors = [], 0

        # 1. 行為測試
        s_ok = r_ok = 0
        total = len(SENTIMENT_CASES) + len(RELEVANCE_CASES)
        for i, (desc, title, content, expected) in enumerate(SENTIMENT_CASES):
            print(f"\r  {name}：行為測試 {i + 1}/{total}", end="", flush=True)
            r = run_sentiment(model, version, title, content)
            behavioral.setdefault(("sentiment", i), {})[name] = r
            s_ok += r["label"] == expected
            secs.append(r["sec"])
            errors += r["error"]
        for i, (desc, title, content, expected) in enumerate(RELEVANCE_CASES):
            print(
                f"\r  {name}：行為測試 {len(SENTIMENT_CASES) + i + 1}/{total}",
                end="",
                flush=True,
            )
            r = run_relevance(model, version, title, content)
            behavioral.setdefault(("relevance", i), {})[name] = r
            r_ok += r["label"] == expected
            if not r["by_rule"]:
                secs.append(r["sec"])
            errors += r["error"]

        print()

        # 2. 真實新聞：相關性＋情緒（情緒不論相關性結果都判斷，方便比較）
        rel_changed = sent_changed = main_to_mention = 0
        for k, n in enumerate(real, 1):
            print(f"\r  {name}：真實新聞 {k}/{len(real)}", end="", flush=True)
            rr = run_relevance(
                model, version, n["title"], n["content"], args.stock, stock_name
            )
            sr = run_sentiment(
                model, version, n["title"], n["content"], args.stock, stock_name
            )
            real_res.setdefault(n["id"], {})[name] = {"rel": rr, "sent": sr}
            rel_changed += rr["label"] != n["relevance"]
            main_to_mention += n["relevance"] == "main" and rr["label"] == "mention"
            sent_changed += sr["label"] != n["llm_sentiment_label"]
            secs += [sr["sec"]] + ([] if rr["by_rule"] else [rr["sec"]])
            errors += rr["error"] + sr["error"]
        if real:
            print()

        row = {
            "組合": name,
            "情緒正確": f"{s_ok}/{len(SENTIMENT_CASES)}",
            "相關性正確": f"{r_ok}/{len(RELEVANCE_CASES)}",
            "真實_情緒改判": f"{sent_changed}/{len(real)}" if real else "—",
            "真實_主角→提及": f"{main_to_mention}/{len(real)}" if real else "—",
            "平均秒數": f"{sum(secs) / len(secs):.1f}" if secs else "—",
            "失敗": errors,
            "總時間": f"{(time.time() - t0) / 60:.1f} 分",
        }
        summary.append(row)
        print("  " + "｜".join(f"{k} {v}" for k, v in row.items()))

    # ---------- 輸出 CSV ----------
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    names = [combo_name(m, v) for m, v in combos]

    path_b = os.path.join(OUT_DIR, f"behavioral_{stamp}.csv")
    with open(
        path_b, "w", newline="", encoding="utf-8-sig"
    ) as f:  # utf-8-sig：Excel 開啟不亂碼
        w = csv.writer(f)
        w.writerow(
            ["任務", "說明", "標題", "預期"]
            + [x for n in names for x in (f"{n}_答案", f"{n}_理由")]
        )
        for task, cases in (
            ("sentiment", SENTIMENT_CASES),
            ("relevance", RELEVANCE_CASES),
        ):
            for i, (desc, title, _, expected) in enumerate(cases):
                res = behavioral[(task, i)]
                cells = []
                for n in names:
                    r = res[n]
                    mark = "✓" if r["label"] == expected else "✗"
                    reason = (f"[{r['subject']}] " if r.get("subject") else "") + (
                        r["reason"] or ""
                    )
                    cells += [f"{mark} {r['label']}", reason]
                w.writerow([task, desc, title, expected] + cells)

    if real:
        path_r = os.path.join(OUT_DIR, f"real_{stamp}.csv")
        with open(path_r, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(
                ["news_id", "日期", "標題", "目前相關性", "目前情緒", "目前理由"]
                + [
                    x
                    for n in names
                    for x in (f"{n}_相關性", f"{n}_情緒", f"{n}_主角與理由")
                ]
            )
            for n in real:
                res = real_res[n["id"]]
                cells = []
                for name in names:
                    rr, sr = res[name]["rel"], res[name]["sent"]
                    reason = (f"[{sr['subject']}] " if sr.get("subject") else "") + (
                        sr["reason"] or ""
                    )
                    cells += [rr["label"], sr["label"], reason]
                w.writerow(
                    [
                        n["id"],
                        n["date"],
                        n["title"],
                        n["relevance"],
                        n["llm_sentiment_label"],
                        n["llm_sentiment_reason"],
                    ]
                    + cells
                )

    # ---------- 總結 ----------
    print("\n========== 總結 ==========")
    for row in summary:
        print("｜".join(f"{k} {v}" for k, v in row.items()))
    print(f"\n行為測試明細：{path_b}")
    if real:
        print(f"真實新聞明細：{path_r}")
    print("\n行為測試答錯的題目：")
    for task, cases in (("sentiment", SENTIMENT_CASES), ("relevance", RELEVANCE_CASES)):
        for i, (desc, _, _, expected) in enumerate(cases):
            wrong = [n for n in names if behavioral[(task, i)][n]["label"] != expected]
            if wrong:
                print(f"  [{task}] {desc}（預期 {expected}）：{', '.join(wrong)}")
