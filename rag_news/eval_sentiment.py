"""
情緒判斷的 baseline 比較：舊版 nlptown（BERT） vs 新版 LLM（預設 Gemma 4＋v3 prompt）

測試資料：test_cases.FAIR_SENTIMENT_CASES（一般測試，18 題，正面、中立、負面各 6 題）
  依類別設計，不參考任何模型的錯誤：競爭對手、否定、轉折、標題誇大、多家公司並列、純事實公告。
  要驗證的主張：新版判斷的是「對指定公司的影響」，舊版只能判斷整篇的語氣。

nlptown 的換算和頁面舊版相同（sentiment.py）：1～5 星的機率 → 期望星等 → −1～+1 → 門檻分成三類；
它不知道指定公司是誰，這正是要比較的差異。

輸出：各模型的正確題數、各類別正確題數、macro-F1、混淆矩陣，以及每題的答案（CSV）。
題目只有 18 題，結果只能說「在這些情況下」，不是準確率。

執行（LLM 需要 Ollama；nlptown 第一次會下載模型）：
    python eval_sentiment.py
    python eval_sentiment.py --llm gemma4:12b:v3 taide-8b:latest:v1     # 模型:版本，可多個
"""

import argparse
import csv
import os
import time
from collections import Counter
from datetime import datetime

import llm_sentiment
import sentiment
from test_cases import FAIR_SENTIMENT_CASES

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare")
LABELS = ["positive", "neutral", "negative"]
LABEL_ZH = {"positive": "正", "neutral": "中", "negative": "負", "ERROR": "錯"}


def run_nlptown(cases):
    texts = [sentiment.build_text(c[4], c[5]) for c in cases]
    return [
        {"label": r["sentiment_label"], "reason": f"分數 {r['sentiment']:+.2f}"}
        for r in sentiment.analyze(texts)
    ]


def run_llm(cases, model, version):
    out = []
    for i, (cat, desc, sid, name, title, content, expected) in enumerate(cases, 1):
        print(f"\r  {model} {version}：{i}/{len(cases)}", end="", flush=True)
        try:
            r = llm_sentiment.classify(
                sid, name, title, content, model=model, version=version
            )
            out.append({"label": r.label, "reason": r.reason})
        except Exception as e:
            out.append({"label": "ERROR", "reason": f"{type(e).__name__}: {e}"[:120]})
    print()
    return out


def macro_f1(gold, pred):
    f1s = []
    for lab in LABELS:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred))
        fp = sum(g != lab and p == lab for g, p in zip(gold, pred))
        fn = sum(g == lab and p != lab for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0)
    return sum(f1s) / len(f1s)


def confusion(gold, pred):
    """列：正確答案，欄：模型答案"""
    c = Counter(zip(gold, pred))
    lines = ["        模型→  " + "  ".join(LABEL_ZH[p] for p in LABELS)]
    for g in LABELS:
        lines.append(
            f"    答案 {LABEL_ZH[g]}     "
            + "  ".join(f"{c[(g, p)]:>2}" for p in LABELS)
        )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="情緒判斷 baseline 比較（nlptown vs LLM）"
    )
    parser.add_argument(
        "--llm", nargs="+", default=["gemma4:12b:v3"], help="模型:prompt 版本，可多個"
    )
    args = parser.parse_args()

    cases = FAIR_SENTIMENT_CASES
    gold = [c[-1] for c in cases]
    t0 = time.time()

    results = {"nlptown": run_nlptown(cases)}
    for spec in args.llm:
        model, version = spec.rsplit(":", 1)
        results[f"{model.split(':')[0]}_{version}"] = run_llm(cases, model, version)

    # ---------- 總結 ----------
    cats = list(dict.fromkeys(c[0] for c in cases))
    print(
        f"\n一般測試 {len(cases)} 題（正 {gold.count('positive')}、中 {gold.count('neutral')}、負 {gold.count('negative')}）"
    )
    header = f"{'模型':<14}{'正確':>6}{'macro-F1':>10}" + "".join(
        f"{c:>8}" for c in cats
    )
    print(header)
    for name, res in results.items():
        pred = [r["label"] for r in res]
        correct = sum(g == p for g, p in zip(gold, pred))
        by_cat = [
            f"{sum(p == g for (c, *_, g), p in zip(cases, pred) if c == cat)}/{sum(c[0] == cat for c in cases)}"
            for cat in cats
        ]
        print(
            f"{name:<14}{correct:>3}/{len(cases):<3}{macro_f1(gold, pred):>9.2f}"
            + "".join(f"{x:>9}" for x in by_cat)
        )

    for name, res in results.items():
        print(f"\n{name} 混淆矩陣：\n{confusion(gold, [r['label'] for r in res])}")

    print("\n答錯的題目：")
    for i, c in enumerate(cases):
        wrong = [
            f"{n}→{LABEL_ZH[results[n][i]['label']]}"
            for n in results
            if results[n][i]["label"] != c[-1]
        ]
        if wrong:
            print(f"  [{c[0]}] {c[1]}（答案 {LABEL_ZH[c[-1]]}）：{'、'.join(wrong)}")

    # ---------- CSV ----------
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"sentiment_baseline_{datetime.now():%Y%m%d_%H%M}.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(
            ["類別", "說明", "公司", "標題", "答案"]
            + [x for n in results for x in (f"{n}_答案", f"{n}_理由")]
        )
        for i, c in enumerate(cases):
            cells = []
            for n in results:
                r = results[n][i]
                cells += [
                    ("✓ " if r["label"] == c[-1] else "✗ ") + r["label"],
                    r["reason"],
                ]
            w.writerow([c[0], c[1], c[3], c[4], c[-1]] + cells)
    print(f"\n明細：{path}\n花費 {time.time() - t0:.0f} 秒")
