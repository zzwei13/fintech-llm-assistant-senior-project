"""
RAG 評估：檢索與生成（不需要人工標註）

讀取 chain.py 產生的快取（cache/），對每個報導焦點計算以下指標。
定位是「用新聞解釋情緒指標」，不是預測，所以評估重點放在：
焦點有沒有解釋指標、證據找得對不對、重點有沒有原文依據。

檢索
  涵蓋率          被挑選焦點的淨貢獻加總 ÷ 指標（焦點能解釋多少指標）
  焦點內聚度      焦點成員（標題＋理由）兩兩相似度的平均；越高代表越像同一件事
  焦點重複度      不同焦點的重點兩兩相似度；太高代表同一件事被拆成兩個焦點
  相關片段比例    證據片段中有提到公司的比例（概念同 RAGAS context precision）
  證據支持率      重點裡的數字，有幾成出現在「檢索到的證據片段」中（概念同 context recall）
                  數字核對比的是整篇原文；這裡比的是檢索結果，用來看檢索有沒有找到對的段落

生成
  數字核對通過率  重點裡的數字都出現在代表新聞原文中的比例
  重點來源分布    reason（判斷理由）／excerpt（原文句子）／title（標題）
  語意支持度      重點和證據句子的最高相似度（bge-m3）；低於門檻的列出來人工檢查
  名稱規則遵守率  14 字以內（prompt 要求 12 字，程式截斷到 14 字）、沒有公司名、沒有利多利空這類判斷
                  （數字不再禁止：實測禁止數字時，模型會用「歷史新高」這類不精確的說法代替「28 個月新高」）
  名稱相關度      名稱和重點的相似度
  LLM 評審        用「另一個模型」（預設 Llama 3.1，不用產生結果的 Gemma，避免自己評自己）判斷：
                    支持度：重點是否被檢索到的證據支持（支持／部分支持／不支持）
                    方向：重點對公司是利多、利空還是中立，和焦點所在欄位（推升／拉低／其他）是否一致；
                          只有「方向相反」（利多 ↔ 利空）算衝突，評審判中立多半是評審比較保守
                    名稱：名稱是否只概括重點，沒有加入重點沒有的事實或形容（例如「28 個月新高」寫成「歷史新高」）
                  概念同 RAGAS faithfulness、ALCE citation；評審模型本身也會出錯，只能當參考
                  （實測 Llama 3.1 偏向判中立、理由和結論矛盾；Qwen 3.5 和人工檢查一致，所以預設用 Qwen）

限制：沒有人工標註，所以這些是「品質指標」，不是準確率。

執行（先用 chain.py 產生快取；Ollama 要開著）：
    python eval_rag.py
    python eval_rag.py --stocks 2330 --judge llama3.1:8b
    python eval_rag.py --no-judge          # 不呼叫評審模型，只算自動指標

輸出：
    compare/eval_rag_<時間>_focus.csv     每個焦點一列（含沒通過的原因，方便人工檢查）
    compare/eval_rag_<時間>_summary.csv   每檔股票一列
"""

import argparse
import csv
import json
import os
import re
import statistics
import time
from datetime import datetime
from itertools import combinations
from typing import Literal

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

import db
import events
from datetime import date

from chain import CACHE_VERSION, cache_path, key_numbers, normalize
from relevance import make_llm
from vectorstore import news_text

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare")
DEFAULT_STOCKS = ["2330", "2317", "2002"]
DEFAULT_END = "2026-10-26"
DEFAULT_JUDGE = "qwen3.5:9b"  # Llama 3.1 實測偏向判中立、理由和結論矛盾，改用 Qwen

DUPLICATE_SIM = 0.80  # 兩個焦點的重點相似度超過這個值，視為可能重複
SUPPORT_MIN_SIM = 0.60  # 語意支持度低於這個值，列為需要人工檢查
NAME_MAX = 14
JUDGMENT_WORDS = ("利多", "利空", "正面", "負面", "看好", "看壞")
KIND_DIRECTION = {"up": "positive", "down": "negative", "neutral": "neutral"}
KIND_ZH = {"up": "推升", "down": "拉低", "neutral": "其他"}


# ---------- LLM 評審 ----------


class Judgment(BaseModel):
    # 先寫理由，再下結論
    reason: str = Field(description="一句話說明判斷依據，60 字以內")
    support: Literal["supported", "partial", "unsupported"] = Field(
        description="重點是否被證據支持：supported 全部有依據、partial 部分有依據、unsupported 沒有依據或和證據矛盾"
    )
    direction: Literal["positive", "neutral", "negative"] = Field(
        description="這個重點對指定公司：positive 利多、neutral 中立、negative 利空"
    )
    name_ok: bool = Field(
        description="名稱是否只概括重點中的內容，沒有加入重點沒有的事實、形容或程度"
    )


JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是嚴格的事實查核員，只輸出 JSON。你會拿到一則關於指定公司的「重點」，以及幾段新聞原文「證據」。\n"
            "1. support：重點中的每個事實和數字，是否都能在證據中找到。"
            "全部找得到是 supported；只有部分找得到是 partial；找不到或和證據矛盾是 unsupported。"
            "只能根據證據判斷，不可使用你自己的知識。\n"
            "2. direction：這個重點描述的事，對指定公司是利多（positive）、利空（negative）還是中立（neutral）。"
            "競爭對手遇到困難，對指定公司是利多。\n"
            "3. name_ok：名稱是這則重點的標題。名稱只概括重點中的內容時為 true；"
            "名稱加入了重點沒有的事實、形容或程度（例如重點寫「28 個月新高」，名稱卻寫「歷史新高」）時為 false。\n"
            "reason 用一句繁體中文說明依據。",
        ),
        (
            "human",
            "指定公司：{stock_name}\n\n名稱：{name}\n\n重點：{point}\n\n證據：\n{evidence}",
        ),
    ]
)


def make_judge(model):
    return JUDGE_PROMPT | make_llm(model).with_structured_output(
        Judgment, method="json_schema"
    )


def judge_focus(judge, stock_name, name, point, evidence, retries=1):
    blocks = (
        "\n".join(f"[{i}] {ev['text']}" for i, ev in enumerate(evidence, 1))
        or "（沒有證據片段）"
    )
    for _ in range(retries + 1):
        try:
            return judge.invoke(
                {
                    "stock_name": stock_name,
                    "name": name,
                    "point": point,
                    "evidence": blocks,
                }
            )
        except Exception as e:  # JSON 格式錯誤、逾時
            err = e
    print(f"  評審失敗（{type(err).__name__}）：{point[:20]}")
    return None


# ---------- 自動指標 ----------


def sentences(text):
    return [
        s.strip()
        for s in re.split(r"(?<=[。！？])|\n", text or "")
        if len(s.strip()) >= 8
    ]


def cohesion(member_sources):
    """焦點成員（標題＋理由）兩兩相似度的平均；只有 1 篇時回傳 None"""
    if len(member_sources) < 2:
        return None
    texts = [
        events.article_text({"title": s["title"], "llm_sentiment_reason": s["reason"]})
        for s in member_sources
    ]
    v = events.embed(texts)
    sims = v @ v.T
    n = len(texts)
    return float((sims.sum() - n) / (n * (n - 1)))  # 去掉對角線（自己和自己）


def number_support(point, text):
    """重點裡的數字有幾成出現在 text；沒有數字時回傳 None"""
    nums = key_numbers(point)
    if not nums:
        return None
    src = normalize(text)
    return sum(x in src for x in nums) / len(nums)


def semantic_support(point, evidence):
    """重點和證據句子的最高相似度"""
    cands = [s for ev in evidence for s in sentences(ev["text"])]
    if not cands:
        return None
    v = events.embed([point] + cands)
    return float((v[1:] @ v[0]).max())


def name_checks(name, stock_name, stock_id):
    """名稱規則：回傳不符合的項目清單"""
    problems = []
    if len(name) > NAME_MAX:
        problems.append(f"超過{NAME_MAX}字")
    if stock_name in name or stock_id in name:
        problems.append("有公司名")
    if any(w in name for w in JUDGMENT_WORDS):
        problems.append("有判斷字眼")
    return problems


def pairwise_max(texts):
    """回傳 [(相似度, i, j)]，由高到低"""
    if len(texts) < 2:
        return []
    v = events.embed(texts)
    sims = v @ v.T
    return sorted(
        ((float(sims[i, j]), i, j) for i, j in combinations(range(len(texts)), 2)),
        reverse=True,
    )


# ---------- 主流程 ----------


def load_cache(stock_id, end):
    path = cache_path(
        stock_id, date.fromisoformat(end)
    )  # 資料指紋要和目前資料庫一致，否則代表快取過期
    if not os.path.exists(path):
        raise SystemExit(
            f"找不到 {path}（沒有快取，或資料已更新），請先執行 python chain.py --stock {stock_id} --end {end}"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_stock(stock_id, end, judge):
    r = load_cache(stock_id, end)
    name = r["stock_name"]
    sources = {s["n"]: s for s in r["sources"]}
    focuses = [(kind, f) for kind in ("up", "down", "neutral") for f in r[kind]]
    rows = []

    for kind, f in focuses:
        members = [sources[n] for n in f["refs"]]
        rep = members[0]  # 代表新聞排在 refs 第一個
        evidence = f["evidence"]
        ev_text = "\n".join(ev["text"] for ev in evidence)
        full_text = rep["title"] + news_text(rep["news_id"])

        row = {
            "stock_id": stock_id,
            "欄位": KIND_ZH[kind],
            "名稱": f["name"],
            "重點": f["point"],
            "重點來源": f["point_source"],
            "篇數": f["n"],
            "淨貢獻": f["net"],
            # 檢索
            "內聚度": cohesion(members),
            "證據片段數": len(evidence),
            "相關片段比例": (
                sum(name in ev["text"] or stock_id in ev["text"] for ev in evidence)
                / len(evidence)
            )
            if evidence
            else None,
            "證據平均相似度": statistics.mean(ev["similarity"] for ev in evidence)
            if evidence
            else None,
            "證據支持率": number_support(f["point"], ev_text),
            # 生成
            "數字核對": number_support(f["point"], full_text),
            "語意支持度": semantic_support(f["point"], evidence),
            "名稱問題": "、".join(name_checks(f["name"], name, stock_id)),
            "名稱相關度": float(
                (lambda v: v[0] @ v[1])(events.embed([f["name"], f["point"]]))
            ),
        }

        if judge is not None:
            j = judge_focus(judge, name, f["name"], f["point"], evidence)
            row["評審支持度"] = j.support if j else "ERROR"
            row["評審方向"] = j.direction if j else "ERROR"
            row["方向一致"] = (j.direction == KIND_DIRECTION[kind]) if j else None
            # 方向相反（利多 ↔ 利空）才是真正的衝突；評審判中立多半是評審比較保守
            row["方向相反"] = (
                ({j.direction, KIND_DIRECTION[kind]} == {"positive", "negative"})
                if j
                else None
            )
            row["評審名稱準確"] = j.name_ok if j else None
            row["評審理由"] = j.reason if j else ""

        # 需要人工檢查的原因
        flags = []
        if row["數字核對"] is not None and row["數字核對"] < 1:
            flags.append("數字未全部出現在原文")
        if row["語意支持度"] is not None and row["語意支持度"] < SUPPORT_MIN_SIM:
            flags.append("和證據相似度低")
        if row["名稱問題"]:
            flags.append("名稱：" + row["名稱問題"])
        if judge is not None and row.get("評審支持度") in ("partial", "unsupported"):
            flags.append(f"評審：{row['評審支持度']}")
        if judge is not None and row.get("評審名稱準確") is False:
            flags.append("評審：名稱不精確")
        if judge is not None and row.get("方向相反"):
            flags.append(f"評審方向相反（{row['評審方向']}）")
        elif judge is not None and row.get("方向一致") is False:
            flags.append(f"評審判 {row['評審方向']}")
        row["需檢查"] = "；".join(flags)
        rows.append(row)

    # 焦點重複度：推升、拉低焦點的重點兩兩比較
    main = [(kind, f) for kind, f in focuses if kind in ("up", "down")]
    pairs = pairwise_max([f["point"] for _, f in main])
    duplicates = [
        (s, main[i][1]["name"], main[j][1]["name"])
        for s, i, j in pairs
        if s >= DUPLICATE_SIM
    ]

    summary = summarize(stock_id, name, r, rows, pairs, duplicates, judge is not None)
    return rows, summary, duplicates


def ratio(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def summarize(stock_id, name, r, rows, pairs, duplicates, judged):
    src = [x["重點來源"] for x in rows]
    s = {
        "stock_id": stock_id,
        "股票": name,
        "情緒指標": round(r["score"], 3) if r["score"] is not None else None,
        "焦點總數": r["n_focuses"],
        "評估焦點數": len(rows),
        # 檢索
        "涵蓋率": r["coverage"],
        "平均內聚度": ratio([x["內聚度"] for x in rows]),
        "最高重複度": pairs[0][0] if pairs else None,
        "可能重複的焦點組數": len(duplicates),
        "相關片段比例": ratio([x["相關片段比例"] for x in rows]),
        "證據支持率": ratio([x["證據支持率"] for x in rows]),
        # 生成
        "數字核對通過率": ratio(
            [x["數字核對"] == 1 for x in rows if x["數字核對"] is not None]
        ),
        "來源_理由": src.count("reason") / len(src) if src else None,
        "來源_原文句子": src.count("excerpt") / len(src) if src else None,
        "來源_標題": src.count("title") / len(src) if src else None,
        "平均語意支持度": ratio([x["語意支持度"] for x in rows]),
        "名稱規則遵守率": ratio([not x["名稱問題"] for x in rows]),
        "平均名稱相關度": ratio([x["名稱相關度"] for x in rows]),
        "需檢查焦點數": sum(bool(x["需檢查"]) for x in rows),
    }
    if judged:
        ok = [x for x in rows if x["評審支持度"] != "ERROR"]
        s["評審_支持"] = (
            sum(x["評審支持度"] == "supported" for x in ok) / len(ok) if ok else None
        )
        s["評審_部分支持"] = (
            sum(x["評審支持度"] == "partial" for x in ok) / len(ok) if ok else None
        )
        s["評審_不支持"] = (
            sum(x["評審支持度"] == "unsupported" for x in ok) / len(ok) if ok else None
        )
        s["評審_方向一致"] = ratio([x["方向一致"] for x in ok])
        s["評審_方向相反"] = ratio([x["方向相反"] for x in ok])
        s["評審_名稱準確"] = ratio([x["評審名稱準確"] for x in ok])
        s["評審失敗"] = len(rows) - len(ok)
    return s


PERCENT_KEYS = ("涵蓋率", "比例", "支持率", "通過率", "來源_", "遵守率", "評審_")


def fmt(key, v):
    """比例顯示成百分比；相似度、指標顯示成小數"""
    if v is None:
        return "—"
    if isinstance(v, float):
        if any(k in key for k in PERCENT_KEYS):
            return f"{v:.0%}"
        return f"{v:+.2f}" if key == "情緒指標" else f"{v:.2f}"
    return str(v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 評估（檢索與生成）")
    parser.add_argument("--stocks", nargs="+", default=DEFAULT_STOCKS)
    parser.add_argument("--end", default=DEFAULT_END, help="chain.py 快取的結束日期")
    parser.add_argument(
        "--judge",
        default=DEFAULT_JUDGE,
        help=f"評審模型（預設 {DEFAULT_JUDGE}），不要和產生結果的模型相同",
    )
    parser.add_argument("--no-judge", action="store_true", help="不呼叫評審模型")
    args = parser.parse_args()

    judge = None if args.no_judge else make_judge(args.judge)
    print(
        f"評估 {args.stocks}（快取 {CACHE_VERSION}、結束日 {args.end}）；評審模型：{'無' if judge is None else args.judge}\n"
    )

    t0 = time.time()
    all_rows, summaries = [], []
    for sid in args.stocks:
        print(f"{sid} {db.get_stock_name(sid)}…", flush=True)
        rows, summary, duplicates = evaluate_stock(sid, args.end, judge)
        all_rows += rows
        summaries.append(summary)
        for sim, a, b in duplicates:
            print(f"  可能重複：{a} ／ {b}（{sim:.2f}）")

    # ---------- 輸出 ----------
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    path_f = os.path.join(OUT_DIR, f"eval_rag_{stamp}_focus.csv")
    path_s = os.path.join(OUT_DIR, f"eval_rag_{stamp}_summary.csv")
    for path, data in ((path_f, all_rows), (path_s, summaries)):
        with open(
            path, "w", newline="", encoding="utf-8-sig"
        ) as f:  # utf-8-sig：Excel 開啟不亂碼
            w = csv.DictWriter(f, list(data[0].keys()))
            w.writeheader()
            w.writerows(data)

    # ---------- 總結 ----------
    keys = [k for k in summaries[0] if k not in ("stock_id",)]
    print("\n========== 總結 ==========")
    print(f"{'指標':<14}" + "".join(f"{s['stock_id']:>10}" for s in summaries))
    for k in keys[1:]:
        print(f"{k:<14}" + "".join(f"{fmt(k, s[k]):>10}" for s in summaries))

    flagged = [x for x in all_rows if x["需檢查"]]
    print(f"\n需要人工檢查的焦點（{len(flagged)} 個）：")
    for x in flagged:
        print(f"  {x['stock_id']} {x['欄位']}【{x['名稱']}】{x['需檢查']}")
    print(f"\n明細：{path_f}\n總表：{path_s}\n花費 {time.time() - t0:.0f} 秒")
