"""
消融實驗：新版 RAG 拿掉一個元件，看每個設計是不是必要的

不做 baseline 比較：舊版（2024）沒有 RAG（30 天摘要全部交給 LLM），沒有可比較的檢索與引用；
中間的 map-reduce 版本是過渡版本，沒有正式使用。

    新版          完整做法：報導焦點（種子檢索分群），推升、拉低各取 3 個
    不分群        直接取 |貢獻| 最大的單篇新聞（推升、拉低各 3 篇）
    不限日期      分群時不限制和種子相隔幾天
    代表用種子    代表新聞用種子，不用 medoid
    片段不優先    證據片段只按相似度排序，不讓有提到公司的片段優先
    不用原文補強  數字核對不通過時，仍直接用理由當重點

指標（全部由程式計算，不呼叫生成模型、不需要人工標註）：
  涵蓋率        挑出的項目淨貢獻加總 ÷ 指標；能解釋多少指標
  方向矛盾      指標偏正面（> 0.2），顯示的項目卻多數是負面（或反過來）；舊版改版的動機
  負面項目比例  顯示的項目中，淨貢獻為負的比例
  重複組數      顯示的項目中，代表新聞（標題＋理由）相似度 ≥ 0.80 的配對數；同一件事佔了好幾個名額
  平均內聚度    焦點成員兩兩相似度的平均（單篇項目不計）
  相關片段比例  證據片段中有提到公司的比例
  數字核對通過率 重點裡的數字都出現在代表新聞原文中的比例

為了不只看單一案例，預設跑 3 檔股票 × 3 個結束日期（每週一次），共 9 個案例取平均。

執行（Ollama 要開著，只用 bge-m3，不用 Gemma）：
    python eval_ablation.py
    python eval_ablation.py --stocks 2330 --ends 2026-10-26

輸出：
    compare/ablation_<時間>_cases.csv    每個案例 × 每種做法一列
    compare/ablation_<時間>_summary.csv  每種做法一列（9 個案例的平均）
"""

import argparse
import csv
import os
import statistics
import time
from datetime import date, datetime

import numpy as np

import db
import events
import trend
from chain import (
    EVIDENCE_K,
    EVIDENCE_KEEP,
    EVIDENCE_PER_NEWS,
    clean_reason,
    grounded,
    make_point,
    normalize,
)
from vectorstore import news_text, search

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare")
DEFAULT_STOCKS = ["2330", "2317", "2002"]
DEFAULT_ENDS = ["2026-10-12", "2026-10-19", "2026-10-26"]

DUPLICATE_SIM = 0.80


# ---------- 把不同做法的結果統一成「項目」 ----------
# 項目 = dict(members, idx, representative, rep_idx, net)，和 events 的事件格式相容


def single(i, arts):
    a = arts[i]
    return {
        "members": [a],
        "idx": [i],
        "representative": a,
        "rep_idx": i,
        "net": a["contribution"],
        "n": 1,
    }


def shown(ranked):
    """顯示在頁面上的項目：推升＋拉低（不含中立）"""
    return ranked["up"] + ranked["down"]


# ---------- 各種做法 ----------


def method_full(arts, sims, **kw):
    return events.rank_events(events.build_events(arts, sims=sims, **kw))


def method_no_cluster(arts, sims):
    """消融：不分群，直接取 |貢獻| 最大的單篇"""
    items = [single(i, arts) for i in range(len(arts))]
    up = sorted([x for x in items if x["net"] > 0], key=lambda x: -x["net"])[
        : events.TOP_EVENTS
    ]
    down = sorted([x for x in items if x["net"] < 0], key=lambda x: x["net"])[
        : events.TOP_EVENTS
    ]
    return {"up": up, "down": down}


def method_seed_rep(arts, sims):
    """消融：代表新聞用種子"""
    evs = events.build_events(arts, sims=sims)
    for e in evs:
        e["representative"] = e["seed"]
        e["rep_idx"] = e["seed_idx"]
    return events.rank_events(evs)


# ---------- 指標 ----------


def evidence_for(item, stock_id, stock_name, start, end, prioritize=True):
    """和 chain.retrieve_evidence 相同；prioritize=False 時不讓提到公司的片段優先（消融用）"""
    rep = item["representative"]
    hits = search(
        events.article_text(rep),
        stock_id,
        k=EVIDENCE_K,
        start_date=int(start.strftime("%Y%m%d")),
        end_date=int(end.strftime("%Y%m%d")),
        relevance=("main",),
        news_ids=[a["id"] for a in item["members"]],
    )
    if prioritize:
        hits = sorted(
            hits,
            key=lambda h: (
                not (stock_name in h[0].page_content or stock_id in h[0].page_content)
            ),
        )
    evidence, per_news = [], {}
    for doc, dist in hits:
        nid = doc.metadata["news_id"]
        if per_news.get(nid, 0) >= EVIDENCE_PER_NEWS:
            continue
        per_news[nid] = per_news.get(nid, 0) + 1
        evidence.append(
            {
                "news_id": nid,
                "text": doc.page_content.split("\n", 1)[-1],
                "similarity": round(1 - dist, 3),
            }
        )
        if len(evidence) == EVIDENCE_KEEP:
            break
    return evidence


def measure(
    ranked,
    arts,
    sims,
    score,
    stock_id,
    stock_name,
    start,
    end,
    prioritize=True,
    use_excerpt=True,
    generation=True,
):
    items = shown(ranked)
    nonzero = [x for x in items if x["net"] != 0]
    neg_share = sum(x["net"] < 0 for x in nonzero) / len(nonzero) if nonzero else None
    pos_share = 1 - neg_share if neg_share is not None else None
    contradiction = None
    if neg_share is not None:
        contradiction = (score > trend.LABEL_BAND and neg_share > 0.5) or (
            score < -trend.LABEL_BAND and pos_share > 0.5
        )

    reps = [x["rep_idx"] for x in items]
    pair_sims = [float(sims[a, b]) for k, a in enumerate(reps) for b in reps[k + 1 :]]
    coh = [
        float(
            (sims[np.ix_(x["idx"], x["idx"])].sum() - len(x["idx"]))
            / (len(x["idx"]) * (len(x["idx"]) - 1))
        )
        for x in items
        if len(x["idx"]) >= 2
    ]

    m = {
        "項目數": len(items),
        "涵蓋率": sum(x["net"] for x in items) / score if score else None,
        "方向矛盾": contradiction,
        "負面項目比例": neg_share,
        "重複組數": sum(s >= DUPLICATE_SIM for s in pair_sims),
        "最高重複度": max(pair_sims) if pair_sims else None,
        "平均內聚度": statistics.mean(coh) if coh else None,
        "相關片段比例": None,
        "數字核對通過率": None,
    }
    if not generation or not items:
        return m

    relevant, total, passed = 0, 0, []
    for x in items:
        ev = evidence_for(x, stock_id, stock_name, start, end, prioritize)
        relevant += sum(stock_name in e["text"] or stock_id in e["text"] for e in ev)
        total += len(ev)
        rep = x["representative"]
        if use_excerpt:
            point, _ = make_point(x, ev, stock_name)
        else:
            point = clean_reason(rep.get("llm_sentiment_reason")) or rep["title"]
        passed.append(grounded(point, normalize(rep["title"] + news_text(rep["id"]))))
    m["相關片段比例"] = relevant / total if total else None
    m["數字核對通過率"] = sum(passed) / len(passed)
    return m


# ---------- 主流程 ----------

ABLATIONS = ["新版", "不分群", "不限日期", "代表用種子", "片段不優先", "不用原文補強"]


def run_case(stock_id, end):
    stats = trend.compute(stock_id, end)
    arts = stats["articles"]
    if stats["score"] is None or len(arts) < trend.MIN_NEWS:
        return None
    name = db.get_stock_name(stock_id)
    sims = events.similarity_matrix(arts)  # 只算一次，所有做法共用
    start, score = stats["start"], stats["score"]
    args = (arts, sims, score, stock_id, name, start, end)

    full = method_full(arts, sims)
    results = {
        "新版": measure(full, *args),
        "不分群": measure(method_no_cluster(arts, sims), *args),
        "不限日期": measure(method_full(arts, sims, max_gap=10**4), *args),
        "代表用種子": measure(method_seed_rep(arts, sims), *args),
        "片段不優先": measure(full, *args, prioritize=False),
        "不用原文補強": measure(full, *args, use_excerpt=False),
    }
    return {"score": score, "n": len(arts), "results": results}


def average(values):
    vals = [float(v) for v in values if v is not None]
    return statistics.mean(vals) if vals else None


PERCENT = ("涵蓋率", "方向矛盾", "比例", "通過率")


def fmt(key, v):
    if v is None:
        return "—"
    if any(k in key for k in PERCENT):
        return f"{v:.0%}"
    if key in ("項目數", "重複組數"):
        return f"{v:.1f}"
    return f"{v:.2f}"


def print_table(title, methods, summary, keys):
    print(f"\n========== {title} ==========")
    print(f"{'做法':<10}" + "".join(f"{k:>10}" for k in keys))
    for mth in methods:
        print(f"{mth:<10}" + "".join(f"{fmt(k, summary[mth][k]):>12}" for k in keys))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="消融實驗")
    parser.add_argument("--stocks", nargs="+", default=DEFAULT_STOCKS)
    parser.add_argument(
        "--ends", nargs="+", default=DEFAULT_ENDS, help="結束日期，可多個"
    )
    args = parser.parse_args()

    t0 = time.time()
    case_rows, collected = [], {}
    for sid in args.stocks:
        for e in args.ends:
            print(f"{sid} 結束日 {e}…", flush=True)
            case = run_case(sid, date.fromisoformat(e))
            if case is None:
                print("  主角新聞不足，略過")
                continue
            for mth, m in case["results"].items():
                collected.setdefault(mth, []).append(m)
                case_rows.append(
                    {
                        "stock_id": sid,
                        "結束日": e,
                        "指標": round(case["score"], 3),
                        "主角新聞": case["n"],
                        "做法": mth,
                        **m,
                    }
                )

    if not collected:
        raise SystemExit("沒有可評估的案例")
    keys = list(next(iter(collected.values()))[0].keys())
    summary = {
        mth: {k: average([m[k] for m in ms]) for k in keys}
        for mth, ms in collected.items()
    }

    # ---------- 輸出 ----------
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    path_c = os.path.join(OUT_DIR, f"ablation_{stamp}_cases.csv")
    path_s = os.path.join(OUT_DIR, f"ablation_{stamp}_summary.csv")
    with open(path_c, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, list(case_rows[0].keys()))
        w.writeheader()
        w.writerows(case_rows)
    with open(path_s, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, ["做法"] + keys)
        w.writeheader()
        for mth in ABLATIONS:
            w.writerow({"做法": mth, **summary[mth]})

    n_cases = len(collected["新版"])
    print(f"\n共 {n_cases} 個案例（數值為平均；方向矛盾 = 發生矛盾的案例比例）")
    print_table(
        "消融實驗（新版拿掉一個元件）",
        ABLATIONS,
        summary,
        [
            "項目數",
            "涵蓋率",
            "方向矛盾",
            "重複組數",
            "最高重複度",
            "平均內聚度",
            "相關片段比例",
            "數字核對通過率",
        ],
    )
    print(f"\n明細：{path_c}\n總表：{path_s}\n花費 {time.time() - t0:.0f} 秒")
