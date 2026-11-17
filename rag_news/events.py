"""
事件分群：把「篇」轉換成「事」，用來說明情緒指標由哪些事件構成

做法：
  1. 取 trend.compute() 的主角新聞（和情緒指標同一批，含每篇的貢獻度）
  2. 每篇用「標題＋TAIDE 理由」以 bge-m3 轉成向量
  3. 以種子檢索建立事件：依時間由新到舊輪流當種子，
     找出尚未歸類、相似度 ≥ TAU 且日期相差 ≤ MAX_GAP_DAYS 的新聞，併成同一事件
  4. （選用，預設關閉）合併重疊的事件：方向相同、日期重疊、成員平均相似度 ≥ MERGE_TAU（每個事件最多合併一次）
     （同一事件的不同報導角度，例如法說會的「獲利與目標價」和「營收與淨利」，
      和種子的相似度可能不到 TAU 而被拆開）
     事件淨貢獻 = 成員貢獻加總（所有事件的淨貢獻加總 = 情緒指標）
  5. 排序：
       推升：淨貢獻 > 0，由大到小
       拉低：淨貢獻 < 0，由小到大
       其他值得注意：成員全為中立，依「時間權重加總」排序（篇數多、越近期越前面）

為什麼要分群：指標的單位是「篇」，使用者理解的單位是「事」；
同一件事常有很多篇報導，單篇貢獻很小（±0.01）又會重複，分群後才看得出哪些事件影響最大。

單獨測試（印出事件清單，用來調整 TAU 和 MAX_GAP_DAYS）：
    python events.py --stock 2330 --end 2026-10-26
    python events.py --stock 2330 --end 2026-10-26 --tau 0.7 --gap 2
"""

import argparse
from datetime import date, datetime

import numpy as np
from langchain_ollama import OllamaEmbeddings

import trend
from vectorstore import EMBEDDING_MODEL

TAU = 0.65  # 相似度門檻：太高會把同一件事拆開，太低會把不同事件合在一起
MAX_GAP_DAYS = (
    5  # 和種子的日期相差超過幾天，就不算同一事件（避免月初、月底的類似報導被合併）
)
# 合併重疊事件的門檻（兩個事件成員的平均相似度）；None = 預設不合併。
# 台積電實測（--merge-report）：要合併的法說會兩個焦點是 0.72，
# 但更高的 0.73 是「德州設廠＋嘉義封裝廠」兩件不同的事，0.69 起也多是泛泛的看好評論；
# 找不到能分開的門檻，所以預設不合併，列為已知限制。要試可以加 --merge-tau 0.72
MERGE_TAU = None
TOP_EVENTS = 3  # 推升、拉低各取幾個事件
TOP_NEUTRAL = 2  # 其他值得注意（中立事件）取幾個
MIN_NEUTRAL_SIZE = 2  # 中立事件至少幾篇報導才列出（單篇的中立新聞多半是瑣事）
REASON_SUFFIXES = ("屬利多", "屬利空", "屬中立")


def article_text(a):
    """事件分群用的文字：標題＋理由（理由去掉「屬利多」這類結論，只留事實）"""
    reason = (a.get("llm_sentiment_reason") or "").strip(" 。")
    for s in REASON_SUFFIXES:
        reason = reason.removesuffix(s).strip(" ，,。")
    return f"{a['title']}\n{reason}"


def embed(texts, batch_size=64):
    """bge-m3 向量，並做 L2 正規化，之後內積就是餘弦相似度"""
    model = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vecs = []
    for i in range(0, len(texts), batch_size):  # 分批送進 bge-m3
        vecs += model.embed_documents(texts[i : i + batch_size])
    m = np.array(vecs, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def representative(idx, net, articles, sims):
    """
    代表新聞：和事件方向一致的成員中，最接近事件中心的一篇
    （和其他成員的平均相似度最高，也就是 medoid），而不是種子本身。
    種子是依時間挑的，可能只是剛好相似的另一件事（例如週報、其他公告）；
    最接近中心的那篇最能代表這群新聞在講什麼。它的理由會作為事件的重點（做法 A）。
    回傳成員的索引。
    """
    sign = 1 if net > 0 else -1 if net < 0 else 0
    cand = [i for i in idx if articles[i]["sentiment"] == sign] or idx
    return max(cand, key=lambda i: float(sims[i, idx].mean()))


def make_event(seed, idx, articles, sims):
    """由成員索引建立事件 dict"""
    members = [articles[i] for i in idx]
    net = sum(a["contribution"] for a in members)
    counts = {
        k: sum(1 for a in members if a["llm_sentiment_label"] == k)
        for k in ("positive", "neutral", "negative")
    }
    rep = representative(idx, net, articles, sims)
    return {
        "idx": idx,  # 成員在 articles 中的索引（合併事件時用）
        "rep_idx": rep,
        "seed_idx": seed,
        "seed": articles[seed],
        "representative": articles[rep],
        "members": members,
        "n": len(members),
        "net": net,  # 淨貢獻
        "weight": sum(a["weight"] for a in members),  # 時間權重加總，中立事件排序用
        "counts": counts,
        "mixed": counts["positive"] > 0 and counts["negative"] > 0,  # 報導看法分歧
        "start": min(a["date"] for a in members),
        "end": max(a["date"] for a in members),
        "similarity": [round(float(sims[seed, i]), 3) for i in idx],
    }


def merge_candidates(events, sims):
    """
    可以合併的事件配對：方向相同（淨貢獻同號）、日期範圍重疊。
    相似度用「平均連結」：兩個事件所有成員兩兩相似度的平均，
    比只看代表新聞穩定，不會因為代表新聞剛好是泛泛的評論文而誤合併。
    回傳 [(平均相似度, 代表新聞相似度, i, j)]，由高到低
    """
    pairs = []
    for i, a in enumerate(events):
        for j in range(i + 1, len(events)):
            b = events[j]
            if a["net"] * b["net"] <= 0:
                continue
            if not (a["start"] <= b["end"] and b["start"] <= a["end"]):
                continue
            link = float(sims[np.ix_(a["idx"], b["idx"])].mean())
            rep = float(sims[a["rep_idx"], b["rep_idx"]])
            pairs.append((link, rep, i, j))
    return sorted(pairs, reverse=True)


def merge_events(events, articles, sims, merge_tau=MERGE_TAU):
    """
    合併重疊的事件：方向相同、日期重疊、平均相似度 ≥ merge_tau。
    每個事件最多合併一次，而且只看原本的事件，不拿合併後的事件再合併——
    否則合併後的代表新聞會變成泛泛的評論文，和什麼都像，最後所有正面事件滾成一大群。
    回傳 (合併後的事件, 合併紀錄)
    """
    used, merged, log = set(), [], []
    for link, rep, i, j in merge_candidates(events, sims):
        if link < merge_tau:
            break
        if i in used or j in used:
            continue
        a, b = events[i], events[j]
        used.update((i, j))
        merged.append(make_event(a["seed_idx"], a["idx"] + b["idx"], articles, sims))
        log.append(
            (
                a["representative"]["title"],
                b["representative"]["title"],
                round(link, 3),
                round(rep, 3),
            )
        )
    rest = [e for k, e in enumerate(events) if k not in used]
    return merged + rest, log


def similarity_matrix(articles):
    """兩兩餘弦相似度（向量已正規化，內積即餘弦相似度）"""
    vecs = embed([article_text(a) for a in articles])
    return vecs @ vecs.T


def build_events(
    articles, tau=TAU, max_gap=MAX_GAP_DAYS, sims=None, merge_tau=MERGE_TAU, log=None
):
    """
    以種子檢索建立事件，再合併重疊的事件。articles：trend.compute()["articles"]
    sims：可傳入已算好的相似度矩陣（調參數時不用重算向量）
    merge_tau：None 代表不合併
    log：傳入 list 時，會把合併紀錄（兩個代表標題、相似度）加進去
    回傳事件清單，每個事件是 dict。
    """
    if not articles:
        return []

    if sims is None:
        sims = similarity_matrix(articles)
    order = sorted(
        range(len(articles)),
        key=lambda i: (
            articles[i]["date"],
            articles[i]["published_at"] or datetime.min,
        ),
        reverse=True,
    )  # 由新到舊（published_at 可能是空值，先比日期）
    assigned = set()
    events = []

    for seed in order:
        if seed in assigned:
            continue
        # 以種子檢索：尚未歸類、相似度夠高、日期相近的新聞
        members_idx = [
            i
            for i in order
            if i not in assigned
            and sims[seed, i] >= tau
            and abs((articles[i]["date"] - articles[seed]["date"]).days) <= max_gap
        ]  # 種子和自己的相似度是 1，一定在裡面
        assigned.update(members_idx)
        events.append(make_event(seed, members_idx, articles, sims))

    if merge_tau is not None:
        events, merged = merge_events(events, articles, sims, merge_tau)
        if log is not None:
            log.extend(merged)
    return events


MIN_NET = 0.005  # |淨貢獻| 小於這個值的事件不列入推升／拉低（例如正負報導互相抵銷、只剩 −0.001）


def rank_events(events, top=TOP_EVENTS, top_neutral=TOP_NEUTRAL):
    """依淨貢獻挑出推升、拉低的事件，以及其他值得注意的中立事件"""
    up = sorted(
        [e for e in events if e["net"] >= MIN_NET], key=lambda e: e["net"], reverse=True
    )
    down = sorted([e for e in events if e["net"] <= -MIN_NET], key=lambda e: e["net"])
    neutral = sorted(
        [
            e
            for e in events
            if e["counts"]["neutral"] == e["n"] and e["n"] >= MIN_NEUTRAL_SIZE
        ],
        key=lambda e: e["weight"],
        reverse=True,
    )
    return {"up": up[:top], "down": down[:top], "neutral": neutral[:top_neutral]}


def analyze_events(
    stock_id, end=None, tau=TAU, max_gap=MAX_GAP_DAYS, merge_tau=MERGE_TAU, log=None
):
    """情緒指標＋事件分群＋排序，回傳 (stats, events, ranked)"""
    stats = trend.compute(stock_id, end)
    events = build_events(stats["articles"], tau, max_gap, merge_tau=merge_tau, log=log)
    ranked = rank_events(events)
    return stats, events, ranked


def coverage(ranked, score):
    """被挑選的推升、拉低事件，淨貢獻加總佔指標的比例"""
    if not score:
        return None
    picked = sum(e["net"] for e in ranked["up"] + ranked["down"])
    return picked / score


# ---------- 單獨測試 ----------


def print_event(k, e, show_members):
    flag = "・報導看法分歧" if e["mixed"] else ""
    c = e["counts"]
    print(
        f"  [{k}] 淨貢獻 {e['net']:+.3f}｜{e['n']} 篇｜{e['start']}～{e['end']}｜"
        f"正 {c['positive']}／中 {c['neutral']}／負 {c['negative']}{flag}"
    )
    print(f"      種子：{e['seed']['title'][:40]}")
    print(f"      代表：{e['representative']['title'][:40]}")
    print(
        f"      代表理由：{(e['representative'].get('llm_sentiment_reason') or '')[:60]}"
    )
    if show_members:
        for a, s in zip(e["members"], e["similarity"]):
            print(
                f"        {s:.2f}  {a['date']}  {a['llm_sentiment_label']:<8} {a['title'][:36]}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="事件分群測試")
    parser.add_argument("--stock", required=True)
    parser.add_argument("--end", help="結束日期 YYYY-MM-DD，預設今天")
    parser.add_argument(
        "--tau", type=float, default=TAU, help=f"相似度門檻，預設 {TAU}"
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=MAX_GAP_DAYS,
        help=f"日期相差上限（天），預設 {MAX_GAP_DAYS}",
    )
    parser.add_argument("--members", action="store_true", help="列出每個事件的成員新聞")
    parser.add_argument(
        "--all", type=int, default=15, help="依 |淨貢獻| 列出前幾個事件，預設 15"
    )
    parser.add_argument(
        "--sweep", action="store_true", help="比較多組 τ 和日期範圍（向量只算一次）"
    )
    parser.add_argument(
        "--merge-tau",
        type=float,
        default=MERGE_TAU,
        help="合併重疊事件的門檻（預設不合併），例如 0.72",
    )
    parser.add_argument(
        "--no-merge", action="store_true", help="不合併重疊事件（比較用）"
    )
    parser.add_argument(
        "--merge-report",
        action="store_true",
        help="列出可合併的候選配對與相似度（不合併），用來選門檻",
    )
    args = parser.parse_args()

    end = date.fromisoformat(args.end) if args.end else None

    if args.sweep:
        stats = trend.compute(args.stock, end)
        sims = similarity_matrix(stats["articles"])
        print(
            f"{args.stock}  情緒指標 {stats['score']:+.3f}｜{len(stats['articles'])} 篇"
        )
        print("   τ    天數  事件數  單篇事件  最大的 5 個事件      涵蓋率")
        for tau in (0.80, 0.75, 0.70, 0.65, 0.60, 0.55):
            for gap in (3, 5, 7):
                evs = build_events(stats["articles"], tau, gap, sims)
                sizes = sorted((e["n"] for e in evs), reverse=True)
                cov = coverage(rank_events(evs), stats["score"])
                print(
                    f"  {tau:.2f}  {gap:>3}  {len(evs):>6}  {sizes.count(1):>8}  {str(sizes[:5]):<20} {cov:>5.0%}"
                )
        raise SystemExit
    if args.merge_report:
        stats = trend.compute(args.stock, end)
        sims = similarity_matrix(stats["articles"])
        evs = build_events(stats["articles"], args.tau, args.gap, sims, merge_tau=None)
        print("平均相似度  代表相似度  事件 A（篇數）＋ 事件 B（篇數）")
        for link, rep, i, j in merge_candidates(evs, sims)[:20]:
            a, b = evs[i], evs[j]
            print(
                f"   {link:.2f}       {rep:.2f}     {a['representative']['title'][:24]}（{a['n']}）"
                f"＋ {b['representative']['title'][:24]}（{b['n']}）"
            )
        raise SystemExit

    merge_tau = None if args.no_merge else args.merge_tau
    merged = []
    stats, events, ranked = analyze_events(
        args.stock, end, args.tau, args.gap, merge_tau, merged
    )

    if stats["score"] is None:
        print("期間內沒有以該公司為主角的新聞")
        raise SystemExit

    sizes = sorted((e["n"] for e in events), reverse=True)
    print(
        f"{args.stock}  {stats['start']} ~ {stats['end']}  τ={args.tau}  日期相差 ≤ {args.gap} 天"
    )
    print(
        f"情緒指標 {stats['score']:+.3f}｜{len(stats['articles'])} 篇 → {len(events)} 個事件"
    )
    print(f"  事件大小前 10：{sizes[:10]}；單篇事件 {sizes.count(1)} 個")
    print(f"  事件淨貢獻加總 {sum(e['net'] for e in events):+.3f}（應等於指標）")
    cov = coverage(ranked, stats["score"])
    if cov is not None:
        print(f"  挑出的推升＋拉低事件涵蓋指標的 {cov:.0%}")
    if merge_tau is not None:
        print(f"\n合併重疊事件（門檻 {merge_tau}）：{len(merged)} 組")
        for t1, t2, link, rsim in merged:
            print(f"  平均 {link:.2f}／代表 {rsim:.2f}  {t1[:26]}  ＋  {t2[:26]}")

    print("\n▲ 推升指標")
    for k, e in enumerate(ranked["up"], 1):
        print_event(k, e, args.members)
    print("\n▼ 拉低指標")
    for k, e in enumerate(ranked["down"], 1):
        print_event(k, e, args.members)
    print("\n○ 其他值得注意（不影響指標）")
    for k, e in enumerate(ranked["neutral"], 1):
        print_event(k, e, args.members)

    print(f"\n依 |淨貢獻| 排序的前 {args.all} 個事件（檢查分群用）：")
    for k, e in enumerate(
        sorted(events, key=lambda e: abs(e["net"]), reverse=True)[: args.all], 1
    ):
        print_event(k, e, args.members)
