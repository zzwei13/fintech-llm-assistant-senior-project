"""
RAG 新聞摘要（新版）：讓重點摘要解釋情緒指標

定位：情緒分析＋RAG 個股新聞摘要，不是問答系統。
每個重點對應一個「報導焦點」（同一事件或同一主題的多篇報導），並標出它對情緒指標的貢獻。

流程：
  1. 情緒指標與每篇貢獻度（trend.compute）
  2. 報導焦點分群與排序（events.analyze_events）
       以種子檢索建立焦點 → 淨貢獻 = 成員貢獻加總 → 推升、拉低各取前 3
  3. 在每個焦點內檢索證據片段（向量檢索，查詢 = 代表新聞的標題＋理由）
  4. 重點文字（不呼叫模型）：
       A. 代表新聞的 LLM 判斷理由，去掉「屬利多」「對公司有正面影響」這類結論
       B. 理由的數字核對不通過 → 改用證據片段中，和理由語意最接近、有提到公司的原文句子
  5. 焦點命名：LLM 呼叫 1 次，替每個焦點取 12 字以內的名稱（也是頁面上的標題）
     摘要：由程式用焦點名稱組成，格式固定、不加入推論

舊版用 map-reduce 讓模型從 8 篇新聞中挑重點；新版改由程式依「對指標的貢獻」挑選、
焦點分群時已經去重，所以不需要 map-reduce，模型呼叫從 9 次降到 1 次。

定位：整理近期新聞作為參考，不預測股價、不提供買賣建議。

單獨測試：
    python chain.py --stock 2330 --end 2026-10-26
    python chain.py --stock 2330 --end 2026-10-26 --refresh   # 忽略快取重新生成
"""

import argparse
import hashlib
import json
import os
import re
import threading
import time
from datetime import date

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, create_model

import events
import trend
from relevance import MODEL, make_llm
from vectorstore import news_text, search

CACHE_VERSION = (
    "v8"  # 程式邏輯或輸出格式改變時才需要更新；資料改變由 data_key() 自動處理
)
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
NUM_CTX = 8192
EVIDENCE_K = 6  # 每個焦點檢索幾段候選片段
EVIDENCE_KEEP = 3  # 保留幾段證據
EVIDENCE_PER_NEWS = 2  # 同一篇新聞最多幾段
SENTENCE_MIN, SENTENCE_MAX = 15, 120  # 做法 B 挑原文句子的長度範圍
POINT_MAX = 70  # 重點文字上限（字）；理由常有兩句以上，太長不適合當重點
SENTENCE_MIN_SIM = 0.6  # 做法 B：原文句子和理由的相似度至少多少才採用
LABEL_ZH = {"positive": "正面", "neutral": "中立", "negative": "負面", None: "未判斷"}

# 理由結尾常見的「結論」子句，當作重點時去掉，只留事實
# 結尾子句含這些詞時視為評語（不是事實）而去掉；「展現／顯示…」是模型自己下的詮釋，原文通常沒有
CONCLUSION_RE = re.compile(
    r"(利多|利空|正面|負面|中立|有利|不利|影響|純事實|純粹|例行|展現|顯示|反映|凸顯)"
)
# 理由在描述「文章」而不是事實，例如「文章僅為預告…」「內文未提供任何…」→ 不適合當重點
META_RE = re.compile(r"^(文章|文中|內文|本文|該文|此文)|未提供|未提及|無具體")


# ---------- 數字核對（沿用舊版） ----------

NUM_RE = re.compile(r"\d+(?:\.\d+)?[%％]?")
YEAR_RE = re.compile(r"^(19|20)\d\d$")


def normalize(text):
    """去掉千分位逗號和空白、全形％轉半形，讓「1 兆 4,942.4」和「1兆4942.4」可以比對。"""
    return re.sub(r"[,，\s]", "", text).replace("％", "%")


def key_numbers(text):
    """可以用來驗證的數字：百分比一律檢查；其他數字至少 2 位，排除年份和個位數"""
    keep = []
    for x in NUM_RE.findall(normalize(text)):
        if x[-1] in "%％":
            keep.append(x[:-1] + "%")
        elif len(x) >= 2 and not YEAR_RE.match(x):
            keep.append(x)
    return keep


def grounded(text, source):
    """text 裡可驗證的數字，是否都出現在 source（已 normalize）裡；沒有數字時視為通過"""
    return all(x in source for x in key_numbers(text))


# ---------- 步驟 3：焦點內檢索證據 ----------


def retrieve_evidence(focus, stock_id, stock_name, start, end):
    """
    在焦點的成員新聞中，以代表新聞的標題＋理由為查詢，檢索最相關的證據片段。
    有提到公司（名稱或代號）的片段優先：盤勢、外資動向這類新聞大部分在講其他個股，
    只看相似度會取到不相關的段落。
    """
    rep = focus["representative"]
    ids = [a["id"] for a in focus["members"]]
    hits = search(
        events.article_text(rep),
        stock_id,
        k=EVIDENCE_K,
        start_date=int(start.strftime("%Y%m%d")),
        end_date=int(end.strftime("%Y%m%d")),
        relevance=("main",),
        news_ids=ids,
    )
    mentions = lambda d: stock_name in d.page_content or stock_id in d.page_content
    hits = sorted(
        hits, key=lambda h: not mentions(h[0])
    )  # 穩定排序：提到公司的在前，其餘維持相似度順序
    evidence, per_news = [], {}
    for doc, dist in hits:
        nid = doc.metadata["news_id"]
        if per_news.get(nid, 0) >= EVIDENCE_PER_NEWS:
            continue
        per_news[nid] = per_news.get(nid, 0) + 1
        body = doc.page_content.split("\n", 1)[-1]  # 去掉片段開頭的「標題：…」
        evidence.append(
            {"news_id": nid, "text": body, "similarity": round(1 - dist, 3)}
        )
        if len(evidence) == EVIDENCE_KEEP:
            break
    return evidence


# ---------- 步驟 4：重點文字（A 為主、B 補強） ----------


def clean_reason(reason):
    """
    理由去掉結尾的結論子句，只留事實。
    例：「外資連9賣…續砍台積電，對台積電有負面影響。」→「外資連9賣…續砍台積電」
    """
    first = re.split(r"[。！？]", (reason or "").strip())[0]  # 只取第一句
    parts = [
        p.strip() for p in re.split(r"[，；;]|,(?!\d)", first) if p.strip()
    ]  # 數字的千分位逗號不切
    while len(parts) > 1 and CONCLUSION_RE.search(parts[-1]):
        parts.pop()
    # 依子句截到 POINT_MAX 字以內，不從句子中間切斷
    kept = parts[:1]
    for p in parts[1:]:
        if len("，".join(kept + [p])) > POINT_MAX:
            break
        kept.append(p)
    return "，".join(kept)


def best_sentence(reason, evidence, stock_name):
    """
    做法 B：從證據片段中，挑和理由語意最接近的原文句子（bge-m3 相似度）。
    只考慮長度適中、有提到公司的句子；相似度低於 SENTENCE_MIN_SIM 就不用，避免選到不相關的句子。
    """
    candidates = []
    for ev in evidence:
        for sent in re.split(r"(?<=[。！？])", ev["text"].replace("\n", "")):
            sent = sent.strip()
            if SENTENCE_MIN <= len(sent) <= SENTENCE_MAX and stock_name in sent:
                candidates.append(sent)
    if not reason or not candidates:
        return None
    vecs = events.embed([reason] + candidates)
    sims = vecs[1:] @ vecs[0]
    k = int(sims.argmax())
    return candidates[k].rstrip("。") if sims[k] >= SENTENCE_MIN_SIM else None


def make_point(focus, evidence, stock_name):
    """
    回傳 (重點文字, 來源)：
      reason   代表新聞的理由（去掉結論）；沒有數字，或數字都出現在原文中
      excerpt  理由的數字核對不通過 → 改用證據中語意最接近理由的原文句子
      title    數字核對不通過、也找不到夠接近的原文句子 → 退回代表新聞的標題
    理由沒有數字時直接使用：「外媒看好」這類主題本來就沒有數字，理由最能說明它被判為正負面的原因。
    理由在描述文章（META_RE）時不使用，改用原文句子（以標題為查詢）或標題。
    """
    rep = focus["representative"]
    reason = clean_reason(rep.get("llm_sentiment_reason"))
    if reason and META_RE.search(reason):
        sentence = best_sentence(rep["title"], evidence, stock_name)
        return (sentence, "excerpt") if sentence else (rep["title"], "title")

    if reason and grounded(reason, normalize(rep["title"] + news_text(rep["id"]))):
        return reason, "reason"

    sentence = best_sentence(reason, evidence, stock_name)
    if sentence:
        return sentence, "excerpt"
    return rep["title"], "title"


# ---------- 步驟 5：焦點命名（呼叫模型 1 次）與摘要（程式組成） ----------
#
# 原本讓模型直接寫 2～3 句摘要，8B 模型常不遵守規則（寫成條列、逐一重述、過長）。
# 改成模型只做小任務：替每個焦點取一個短名稱；摘要由程式用名稱組成，格式固定、不會加入推論。
# 名稱也是頁面上每個「報導焦點」的標題。


def names_schema(n):
    """
    動態建立有 n 個固定欄位的 schema：focus_1、focus_2…
    用清單＋編號時，8B 模型會跳號、錯位（第 3 個焦點拿到第 4 個的名稱）；
    固定欄位由 JSON schema 強制，每個焦點一定有自己的名稱，順序也不會亂。
    """
    # 不在 schema 裡設 max_length：Gemma 4 遇到含 maxLength 的 schema 會回傳空白；長度改由程式截斷
    fields = {
        f"focus_{i}": (str, Field(description=f"焦點 {i} 的名稱，12 字以內"))
        for i in range(1, n + 1)
    }
    return create_model("FocusNames", **fields)


NAME_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是財經新聞編輯，只輸出 JSON。以下是一檔股票近期的幾個報導焦點，每個焦點有編號、重點和代表新聞標題。\n"
            "請為每個焦點取一個名稱。\n"
            "規則：\n"
            "1. 名稱是 12 字以內的名詞片語，概括這群報導在講什麼，例如「法說會財報亮眼」「外資連續賣超」。\n"
            "   不要寫公司名稱，不要照抄新聞標題。\n"
            "2. 只能根據提供的內容命名，不可加入其他資訊。\n"
            "3. 不要加入「利多」「利空」這類判斷；需要精確描述時可以保留重點中的數字（例如「營收創28個月新高」）。\n"
            "   只能用重點裡有的事實，不要加入重點沒有的形容或程度，例如重點沒寫「歷史」，名稱就不能寫「歷史新高」。\n"
            "4. focus_1 對應焦點 1、focus_2 對應焦點 2，依此類推。使用繁體中文。",
        ),
        ("human", "指定公司：{stock_name}\n\n{focuses}"),
    ]
)

NAME_MAX = 14  # 名稱最多幾個字（prompt 要求 12 字，留一點彈性）
TITLE_SPLIT = re.compile(r"[！？!?，,、：:；;｜|》〈〉「」『』\s]+|\s-\s")


def title_name(title, stock_name):
    """命名失敗時的備案：標題依標點切開，取第一段夠長的（去掉公司名）"""
    for part in TITLE_SPLIT.split(title):
        part = part.replace(stock_name, "").strip(" 的")
        if len(part) >= 4:
            return part[:NAME_MAX]
    return title[:NAME_MAX]


def parse_names(raw, n):
    """結構化輸出解析失敗時，自己從模型的原始回覆中找 JSON 物件"""
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    return {i: str(data.get(f"focus_{i}") or "") for i in range(1, n + 1)}


def name_focuses(stock_name, focuses, retries=1):
    """呼叫 1 次（失敗再試 1 次），替每個焦點取名稱；仍失敗時，用代表新聞標題的第一段"""
    n = len(focuses)
    blocks = "\n".join(
        f"焦點 {i}\n  重點：{f['point']}\n  代表新聞：{f['title']}"
        for i, f in enumerate(focuses, 1)
    )
    llm = make_llm(
        MODEL, num_ctx=NUM_CTX, num_predict=1024
    )  # 有思考模式的模型（Gemma 4）會關掉思考
    # include_raw：解析失敗時不丟例外，回傳原始回覆，方便自己解析與除錯
    chain = NAME_PROMPT | llm.with_structured_output(
        names_schema(n), method="json_schema", include_raw=True
    )
    names = {}
    for attempt in range(retries + 1):
        try:
            out = with_timer(
                "焦點命名",
                lambda: chain.invoke({"stock_name": stock_name, "focuses": blocks}),
            )
        except Exception as e:
            print(f"焦點命名失敗（{type(e).__name__}）：{e}")
            continue
        if out["parsed"] is not None:
            names = {i: getattr(out["parsed"], f"focus_{i}") for i in range(1, n + 1)}
            break
        raw = out["raw"].content
        names = parse_names(raw, n)
        if any(names.values()):
            break
        extra = list(getattr(out["raw"], "additional_kwargs", {}).keys())
        print(
            f"焦點命名第 {attempt + 1} 次解析失敗：回覆 {len(raw or '')} 字 {raw[:80]!r}，其他欄位 {extra}"
        )
    for i, f in enumerate(focuses, 1):
        name = (
            (names.get(i) or "").strip(" 「」\"'").replace(stock_name, "").strip(" 的")
        )  # 名稱裡的公司名去掉
        f["name"] = name[:NAME_MAX] if name else title_name(f["title"], stock_name)


TENDENCY = {"positive": "偏正面", "neutral": "中立", "negative": "偏負面"}


def compose_summary(stock_name, stats, up, down):
    """用焦點名稱組成摘要（不呼叫模型）"""
    c = stats["counts"]
    quote = lambda fs: "、".join(f"「{f['name']}」" for f in fs)
    text = (
        f"近 {trend_days(stats)} 天{stock_name}的新聞{TENDENCY[stats['label']]}（{stats['score']:+.2f}），"
        f"主角新聞 {stats['n_30']} 篇：正面 {c['positive']}、中立 {c['neutral']}、負面 {c['negative']}。"
    )
    if up:
        text += f"推升指標的主要是{quote(up)}；"
    text += f"需要留意的有{quote(down)}。" if down else "期間內沒有明顯的負面報導焦點。"
    return text


def trend_days(stats):
    return (stats["end"] - stats["start"]).days + 1


def format_stats(stats):
    c = stats["counts"]
    return (
        f"以該公司為主角的新聞 {stats['n_30']} 篇：正面 {c['positive']}、中立 {c['neutral']}、"
        f"負面 {c['negative']}；加權情緒分數 {stats['score']:+.2f}（{LABEL_ZH[stats['label']]}）"
    )


def with_timer(label, fn):
    """執行 fn()，同時每秒在同一行更新經過的秒數，讓等待時看得到程式還在跑"""
    done = threading.Event()
    t0 = time.time()

    def tick():
        while not done.wait(1):
            print(f"\r  {label}… {time.time() - t0:.0f} 秒", end="", flush=True)

    thread = threading.Thread(target=tick, daemon=True)
    thread.start()
    try:
        return fn()
    finally:
        done.set()
        thread.join()
        print(f"\r  {label}完成，{time.time() - t0:.0f} 秒")


# ---------- 主流程 ----------


def data_key(stats):
    """
    快取的資料指紋：主角新聞的 id、情緒、理由，加上模型與分群參數，取 md5 前 8 碼。
    重新判斷相關性或情緒（換模型、改 prompt）、新增新聞、調整參數時，指紋就會改變，自動重新計算，
    不用再手動改 CACHE_VERSION。
    """
    items = sorted(
        (a["id"], a["llm_sentiment_label"], a["llm_sentiment_reason"] or "")
        for a in stats["articles"]
    )
    params = [
        MODEL,
        events.TAU,
        events.MAX_GAP_DAYS,
        events.MERGE_TAU,
        events.MIN_NET,
        events.TOP_EVENTS,
    ]
    raw = json.dumps([items, params], ensure_ascii=False, default=str)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]


def cache_path(stock_id, end, stats=None):
    """快取檔案路徑：股票_結束日_程式版本_資料指紋.json（stats 沒傳時會查資料庫）"""
    stats = stats or trend.compute(stock_id, end)
    return os.path.join(
        CACHE_DIR, f"{stock_id}_{end}_{CACHE_VERSION}_{data_key(stats)}.json"
    )


def analyze(stock_id, stock_name, end=None, refresh=False):
    end = end or date.today()
    cache_path_ = cache_path(stock_id, end)
    if not refresh and os.path.exists(cache_path_):
        with open(cache_path_, encoding="utf-8") as f:
            return json.load(f)

    # 步驟 1、2：情緒指標、報導焦點分群與排序
    print("報導焦點分群…", flush=True)
    stats, all_events, ranked = events.analyze_events(stock_id, end)
    start = stats["start"]

    result = {
        "stock_id": stock_id,
        "stock_name": stock_name,
        "start": str(start),
        "end": str(end),
        "score": stats["score"],
        "label": stats["label"],
        "enough": stats["enough"],
        "n_focuses": len(all_events),
        "coverage": None,
        "up": [],
        "down": [],
        "neutral": [],
        "sources": [],
        "summary": "期間內沒有以該公司為主角的新聞。",
    }
    if stats["score"] is None:
        return result

    # 步驟 3、4：每個焦點檢索證據、產生重點；引用新聞依焦點分組編號
    seen = {}  # news_id -> 引用編號
    for kind in ("up", "down", "neutral"):
        for focus in ranked[kind]:
            evidence = retrieve_evidence(focus, stock_id, stock_name, start, end)
            point, point_source = make_point(focus, evidence, stock_name)
            refs = []
            # 代表新聞排第一，其餘成員依時間由新到舊
            members = [focus["representative"]] + [
                a for a in focus["members"] if a is not focus["representative"]
            ]
            for a in members:
                if a["id"] not in seen:
                    seen[a["id"]] = len(result["sources"]) + 1
                    result["sources"].append(
                        {
                            "n": seen[a["id"]],
                            "news_id": a["id"],
                            "title": a["title"],
                            "url": a["url"],
                            "date": str(a["date"]),
                            "source": a["source"],
                            "label": a["llm_sentiment_label"],
                            "reason": a["llm_sentiment_reason"],
                            "contribution": round(a["contribution"], 4),
                            "focus": kind,
                        }
                    )
                refs.append(seen[a["id"]])
            result[kind].append(
                {
                    "name": None,  # 步驟 5 由模型命名
                    "title": focus["representative"]["title"],
                    "point": point,
                    "point_source": point_source,  # reason／excerpt／reason_plain／title，方便檢查重點來自哪裡
                    "net": round(focus["net"], 4),
                    "n": focus["n"],
                    "start": str(focus["start"]),
                    "end": str(focus["end"]),
                    "counts": focus["counts"],
                    "mixed": focus["mixed"],
                    "refs": refs,
                    "evidence": evidence,
                }
            )
    result["coverage"] = events.coverage(ranked, stats["score"])

    # 步驟 5：焦點命名（呼叫 1 次）＋摘要（程式組成）
    focuses = result["up"] + result["down"] + result["neutral"]
    if focuses:
        name_focuses(stock_name, focuses)
    result["summary"] = compose_summary(stock_name, stats, result["up"], result["down"])

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path_, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


# ---------- 單獨測試 ----------

if __name__ == "__main__":
    import db

    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", required=True)
    parser.add_argument("--end", help="結束日期 YYYY-MM-DD，預設今天")
    parser.add_argument("--refresh", action="store_true", help="忽略快取重新生成")
    parser.add_argument(
        "--evidence", action="store_true", help="列出每個焦點的證據片段"
    )
    args = parser.parse_args()

    name = db.get_stock_name(args.stock)
    end = date.fromisoformat(args.end) if args.end else None
    t0 = time.time()
    r = analyze(args.stock, name, end, refresh=args.refresh)

    print(
        f"\n{name}（{args.stock}）{r['start']} ~ {r['end']}，花費 {time.time() - t0:.0f} 秒"
    )
    if r["score"] is None:
        print(r["summary"])
        raise SystemExit
    cov = f"{r['coverage']:.0%}" if r["coverage"] is not None else "—"
    print(
        f"情緒指標 {r['score']:+.2f}（{LABEL_ZH[r['label']]}）｜{r['n_focuses']} 個報導焦點｜涵蓋指標 {cov}"
    )
    print(f"\n摘要：{r['summary']}")

    for kind, title in (
        ("up", "▲ 推升指標"),
        ("down", "▼ 拉低指標"),
        ("neutral", "○ 其他值得注意"),
    ):
        if not r[kind]:
            continue
        print(f"\n{title}")
        for f in r[kind]:
            flag = "・報導看法分歧" if f["mixed"] else ""
            print(f"  【{f['name']}】{f['point']}")
            print(
                f"    貢獻 {f['net']:+.3f}・{f['n']} 篇（{f['start']}～{f['end']}）{flag}"
                f"・重點來源：{f['point_source']}・引用 {f['refs'][:5]}"
            )
            if args.evidence:
                for ev in f["evidence"]:
                    print(f"      [{ev['similarity']:.2f}] {ev['text'][:70]}")

    print(f"\n引用新聞：{len(r['sources'])} 篇")
