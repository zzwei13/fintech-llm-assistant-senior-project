"""
nlptown 情緒分析（ingest 階段 3）

模型：nlptown/bert-base-multilingual-uncased-sentiment
  多語言 BERT，以商品評論微調，輸出 1～5 星的機率（softmax）。

換算方式：
  期望星等  E = Σ k·p_k              （k = 1～5）
  情緒分數  s = (E − 3) / 2          （-1 最負面 ～ +1 最正面）
  三分類    s > 0.2 正面、s < -0.2 負面，其餘中立（標籤由分數決定，方向一定一致）
            ※ 不用「p1+p2 / p3 / p4+p5 取最大」：中立只有一格容易被低估，
              且沒有加權，可能出現「分數為負、標籤卻是正面」的矛盾
  五個機率存進 star_probs，之後可重算或換方法，不必重跑模型。

輸入：內文開頭兩段，不含標題（截斷到 512 token）
處理範圍：relevance 為 main 或 mention 的新聞（unrelated 不算）
  ※ 趨勢圖與情緒指標只計入 main，mention 的分數供新聞卡片顯示

單獨測試：
    python sentiment.py
"""

import torch
from transformers import pipeline

MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"
NEUTRAL_BAND = 0.2  # |分數| ≤ 0.2 視為中立（約等於期望星等 2.6～3.4）

_pipe = None


def get_pipe():
    """第一次呼叫時才載入模型"""
    global _pipe
    if _pipe is None:
        if torch.backends.mps.is_available():
            device = "mps"  # gpu
        else:
            device = -1  # cpu
        _pipe = pipeline(
            "text-classification",
            model=MODEL,
            top_k=None,  # 回傳 5 個星等的機率，而不只最高的那個
            truncation=True,  # 輸入太長時，自動把超過的部分截掉
            max_length=512,
            device=device,
        )
    return _pipe


def build_text(title, content):
    """
    開頭兩段（不含標題）
    標題常為了吸引點閱而誇大或用反問句，實測 20 篇：拿掉標題後平均分數 -0.38 → -0.22，
    5 篇標籤改變，偏負的情況減輕。title 參數保留，方便呼叫端不用改。
    """
    paragraphs = []
    for p in content.split("\n"):
        if p.strip():  # 防止內文裡有連續換行
            paragraphs.append(p)
    return "".join(paragraphs[:2])


def to_score(probs):
    """
    probs：[p1, p2, p3, p4, p5]
    回傳 (情緒分數 -1～+1, 三分類標籤)
    """
    expected = 0
    for k, p in enumerate(probs, 1):
        expected += k * p

    score = (expected - 3) / 2  # 轉成範圍-1～+1

    # 標籤由分數決定，和分數的方向一定一致
    if score > NEUTRAL_BAND:
        label = "positive"
    elif score < -NEUTRAL_BAND:
        label = "negative"
    else:
        label = "neutral"
    return score, label


def analyze(texts, batch_size=16):
    """
    批次分析多篇文字，回傳 list of dict：
        {"star_probs": [p1..p5], "sentiment": s, "sentiment_label": label}
    """
    pipe = get_pipe()  # 取得 pipeline 物件
    outputs = pipe(texts, batch_size=batch_size)  # 呼叫它
    results = []
    for out in outputs:
        # out 例：[{"label": "4 stars", "score": 0.41}, {"label": "5 stars", ...}, ...]（順序不固定）
        by_star = {}
        for d in out:
            star = int(d["label"][0])
            by_star[star] = d["score"]
        probs = []
        for k in range(1, 6):  # 按照星等排序
            probs.append(round(by_star[k], 4))
        score, label = to_score(probs)  # 期望星等,標籤
        results.append(
            {
                "star_probs": probs,  # list
                "sentiment": round(score, 4),
                "sentiment_label": label,
            }
        )
    return results


if __name__ == "__main__":
    tests = [
        (
            "台積電第三季獲利創新高 法人調升目標價",
            "台積電公布第三季財報，稅後純益年增五成，優於市場預期。",
        ),
        (
            "台積電跌破2400元 外資連續賣超",
            "外資本周大舉賣超台積電，股價跌破2400元，市場擔憂AI需求降溫。",
        ),
        (
            "台積電將於下周四舉行法說會",
            "台積電公告將於10月16日舉行法人說明會，說明第三季營運成果。",
        ),
    ]
    texts = [build_text(t, c) for t, c in tests]
    for (title, _), r in zip(tests, analyze(texts)):
        probs = " ".join(f"{p:.2f}" for p in r["star_probs"])
        print(f"{r['sentiment']:+.2f} {r['sentiment_label']:<9} [{probs}] | {title}")
