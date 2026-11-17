"""
新聞情緒指標與 30 天趨勢

資料：relevance = main 且已有 TAIDE 情緒（llm_sentiment_label）的新聞
換算：positive +1、neutral 0、negative −1

指標：
  加權分數  近 30 天所有新聞的加權平均，權重 w = 0.5^(距今天數 / 7)
           ——7 天前的新聞權重剩一半，越新的新聞影響越大
  近 7 天 vs 近 30 天  簡單平均，用來看最近是否轉好或轉壞
  標籤      分數 > 0.2 正面、< −0.2 負面，其餘中立
  篇數 < MIN_NEWS 時標示「資料不足」

跨來源重複報導（同一事件被不同媒體各報一次）不去除：
  報導量本身反映事件受關注的程度，投資人接觸到的訊息量也較多。
  （同一來源內的近似重複已在爬蟲階段以 shingle Jaccard 去除）

單獨測試：
    python trend.py --stock 2330
"""

import argparse
from datetime import date, timedelta

import db

SCORE = {"negative": -1, "neutral": 0, "positive": 1}
HALF_LIFE = 7  # 半衰期（天）
DAYS = 30
LABEL_BAND = 0.2
MIN_NEWS = 5


def to_label(score):
    if score > LABEL_BAND:
        return "positive"
    if score < -LABEL_BAND:
        return "negative"
    return "neutral"


def compute(stock_id, end=None, days=DAYS):
    """
    回傳 dict：
      score, label, avg_7, avg_30, n_7, n_30, counts, enough, daily, start, end
      articles：每篇主角新聞（含 id、title、理由）加上 sentiment、weight、contribution，依 |貢獻| 由大到小
      daily：[{"date", "avg", "n"}]，每一天都有一筆，沒有新聞的日子 avg 為 None
    """
    end = end or date.today()
    start = end - timedelta(days=days - 1)
    rows = db.fetch_main_sentiment(stock_id, start, end)

    weighted, weights = 0.0, 0.0
    by_day = {start + timedelta(d): [] for d in range(days)}
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    articles = []  # 每篇新聞的權重與貢獻，供 events.py 建立事件

    for r in rows:
        s = SCORE[r["llm_sentiment_label"]]
        age = (end - r["date"]).days
        w = 0.5 ** (age / HALF_LIFE)
        weighted += w * s
        weights += w
        by_day[r["date"]].append(s)
        counts[r["llm_sentiment_label"]] += 1
        articles.append({**r, "sentiment": s, "weight": w})

    # 貢獻 c_i = s_i·w_i ／ Σw：所有貢獻加總剛好等於指標，中立新聞的貢獻為 0
    for a in articles:
        a["contribution"] = a["sentiment"] * a["weight"] / weights
    articles.sort(key=lambda a: abs(a["contribution"]), reverse=True)

    recent = [s for d, ss in by_day.items() if (end - d).days < 7 for s in ss]
    all_scores = [s for ss in by_day.values() for s in ss]

    score = weighted / weights if weights else None
    return {
        "articles": articles,  # 依 |貢獻| 由大到小
        "score": score,
        "label": to_label(score) if score is not None else None,
        "avg_7": sum(recent) / len(recent) if recent else None,
        "avg_30": sum(all_scores) / len(all_scores) if all_scores else None,
        "n_7": len(recent),
        "n_30": len(all_scores),
        "counts": counts,
        "enough": len(all_scores) >= MIN_NEWS,
        "daily": [
            {"date": d, "avg": sum(ss) / len(ss) if ss else None, "n": len(ss)}
            for d, ss in sorted(by_day.items())
        ],
        "start": start,
        "end": end,
    }


def plot_html(result, stock_name="", height=None, show_title=True):
    """30 天趨勢圖：長條＝每日篇數（右軸）、折線＝每日平均情緒（左軸），回傳可嵌入頁面的 HTML 片段。"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    days = [d["date"] for d in result["daily"]]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=days,
            y=[d["n"] for d in result["daily"]],
            name="新聞篇數",
            marker_color="rgba(150,150,150,0.35)",
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=days,
            y=[d["avg"] for d in result["daily"]],
            name="每日平均情緒",
            mode="lines+markers",
            connectgaps=False,
        ),  # 沒有新聞的日子留空，不畫線
        secondary_y=False,
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(
        title=f"{stock_name} 近 {DAYS} 天新聞情緒" if show_title else None,
        height=height,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=40, r=40, t=50 if show_title else 10, b=40),
    )
    fig.update_yaxes(
        title_text="情緒（−1 負面 ～ +1 正面）", range=[-1.1, 1.1], secondary_y=False
    )
    # 右軸（篇數）用整數刻度；不設定時 plotly 會對齊左軸的格線，出現 28.13、21.44 這種小數
    max_n = max((d["n"] for d in result["daily"]), default=0) or 1
    step = next(s for s in (1, 2, 5, 10, 20, 50, 100, 200, 500) if max_n / s <= 5)
    fig.update_yaxes(
        title_text="篇數",
        range=[0, max_n * 1.15],
        tickmode="linear",
        tick0=0,
        dtick=step,
        showgrid=False,
        secondary_y=True,
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", required=True)
    parser.add_argument("--end", help="結束日期 YYYY-MM-DD，預設今天")
    parser.add_argument("--html", help="把趨勢圖存成 HTML 檔，方便用瀏覽器檢查")
    args = parser.parse_args()

    end = date.fromisoformat(args.end) if args.end else None
    r = compute(args.stock, end)
    fmt = lambda x: "—" if x is None else f"{x:+.2f}"
    print(f"{args.stock}  {r['start']} ~ {r['end']}")
    print(
        f"  加權分數 {fmt(r['score'])}（{r['label']}）  資料{'足夠' if r['enough'] else '不足'}"
    )
    print(
        f"  近 7 天 {fmt(r['avg_7'])}（{r['n_7']} 篇）  近 30 天 {fmt(r['avg_30'])}（{r['n_30']} 篇）"
    )
    print(
        f"  正面 {r['counts']['positive']}、中立 {r['counts']['neutral']}、負面 {r['counts']['negative']}"
    )
    for d in r["daily"]:
        if d["n"]:
            print(f"  {d['date']}  {fmt(d['avg'])}  {'█' * d['n']}")

    # 貢獻度檢查：加總應等於加權分數
    if r["articles"]:
        total = sum(a["contribution"] for a in r["articles"])
        print(f"\n貢獻加總 {total:+.4f}（加權分數 {fmt(r['score'])}）")
        print("貢獻最大的 10 篇：")
        for a in r["articles"][:10]:
            print(
                f"  {a['contribution']:+.3f}  {a['date']}  {a['llm_sentiment_label']:<8} {a['title'][:30]}"
            )

    if args.html:
        with open(args.html, "w", encoding="utf-8") as f:
            f.write(
                f"<html><head><meta charset='utf-8'></head><body>{plot_html(r, args.stock)}</body></html>"
            )
        print(f"\n趨勢圖已存到 {args.html}")
