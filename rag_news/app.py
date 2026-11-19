"""
Flask 頁面：新聞情緒指標、30 天趨勢、指標的組成（RAG 報導焦點）

    python app.py
    瀏覽器打開 http://127.0.0.1:5001/?stock=2330
    指定結束日期：http://127.0.0.1:5001/?stock=2330&end=2026-10-26

三個區塊：
  1. 新聞情緒指標：LLM 判斷的正面／中立／負面，時間衰減加權（trend.compute）
  2. 30 天趨勢：每日平均情緒（折線）＋每日篇數（長條）（trend.plot_html）
  3. 指標的組成：推升／拉低指標的報導焦點、貢獻、摘要與引用新聞（chain.analyze）

日期區間預設以「該股票最新一篇新聞的日期」為結束日，避免今天還沒有新聞時整頁空白。
資料由 python ingest.py update 在背景更新（可排程每天執行），頁面只讀已算好的結果，並顯示更新時間。
RAG 分析第一次產生約需 20 秒，之後讀 cache/；展示前可先執行 python chain.py 預先產生。
"""

import json
import os
from datetime import date

from flask import Flask, abort, render_template, request

import chain
import relevance
import db
import trend

app = Flask(__name__)

LABEL_ZH = {"positive": "正面", "neutral": "中立", "negative": "負面", None: "—"}
TENDENCY = {"positive": "偏正面", "neutral": "中立", "negative": "偏負面", None: "—"}
SOURCE_ZH = {"ltn": "自由時報", "cnyes": "鉅亨網"}
UPDATE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "cache", "last_update.json"
)


def last_update(stock_id):
    """ingest.py update 寫下的更新紀錄；沒有紀錄（例如資料是手動匯入的）時回傳 None"""
    try:
        with open(UPDATE_FILE, encoding="utf-8") as f:
            return json.load(f).get(stock_id)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


@app.route("/")
def index():
    stocks = db.list_stocks_with_news()
    if not stocks:
        return "資料庫裡還沒有新聞，請先執行 python ingest.py crawl"

    stock_id = request.args.get("stock") or stocks[0]["stock_id"]
    stock_name = db.get_stock_name(stock_id)
    if stock_name is None:
        abort(404)

    end = request.args.get("end")
    end = date.fromisoformat(end) if end else db.latest_news_date(stock_id)
    stats = trend.compute(stock_id, end)
    chart = trend.plot_html(stats, stock_name, height=300, show_title=False)
    analysis = chain.analyze(stock_id, stock_name, end)

    # 貢獻長條的比例：以最大的 |淨貢獻| 為 100%
    focuses = analysis["up"] + analysis["down"]
    max_net = max((abs(f["net"]) for f in focuses), default=0) or 1
    sources = {s["n"]: s for s in analysis["sources"]}

    return render_template(
        "index.html",
        stocks=stocks,
        stock_id=stock_id,
        stock_name=stock_name,
        stats=stats,
        chart=chart,
        analysis=analysis,
        max_net=max_net,
        sources=sources,
        label_zh=LABEL_ZH,
        tendency=TENDENCY,
        source_zh=SOURCE_ZH,
        min_news=trend.MIN_NEWS,
        updated=last_update(stock_id),
        model_name=relevance.MODEL.split(":")[0],
    )


if __name__ == "__main__":
    # Mac 的 5000 port 常被 AirPlay 接收器佔用，所以用 5001
    app.run(debug=True, port=5001)
