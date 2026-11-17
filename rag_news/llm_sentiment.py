"""
TAIDE 目標導向情緒判斷（ingest 階段 3b）

只處理 relevance = main 的新聞，判斷「對指定公司而言」是利多、利空還是中立。
作為頁面上的主要情緒指標；nlptown（sentiment.py）保留為 baseline。

為什麼不只用 nlptown（實測 20 篇）：
  ① 被標題帶偏：標題常為吸引點閱而誇大或用反問句（「台股明天免驚？」）
  ② 看不懂財經語意：「止跌」「反彈」「翻紅」仍被判為負面
  ③ 沒有對象：只能判斷整篇語氣，無法判斷對某家公司是利多或利空

做法（沿用相關性判斷的經驗）：
  - 先寫理由，再下結論
  - 輸入：標題 + 開頭兩段 + 提到該公司的句子；prompt 註明以內文事實為準
  - JSON schema 約束輸出、temperature 0

單獨測試：
    python llm_sentiment.py
"""

import os
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from relevance import MODEL, build_inputs, find_mentions, make_llm

# prompt 版本：v1 原版；v2 先寫 subject（新聞主要在講誰）、修正競爭對手與股價的規則（compare_models.py 比較用）；
#             v3 = v1 ＋ 分析文依主要論點判斷（目前使用）
PROMPT_VERSION = os.getenv("SENTIMENT_PROMPT", "v3")
SCORE = {"negative": -1, "neutral": 0, "positive": 1}  # 換算成數值，供趨勢圖計算平均


class LLMSentiment(BaseModel):
    # 先寫理由，再下結論
    reason: str = Field(
        description="一句話寫出影響判斷的具體事實（主詞＋發生什麼事），盡量包含新聞中的數字，80 字以內"
    )
    label: Literal["positive", "neutral", "negative"] = Field(
        description="對指定公司而言：positive 利多、neutral 中立、negative 利空"
    )


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是財經新聞分析師，只輸出 JSON。判斷這篇新聞對「指定公司」是利多、利空還是中立。\n"
            "- positive 利多：營收或獲利成長、接獲訂單、法人調升評等或目標價、股價上漲或止跌回升、新產品或擴產順利\n"
            "- negative 利空：營收或獲利衰退、訂單流失、法人調降評等、股價下跌、訴訟或監管、競爭對手威脅\n"
            "- neutral 中立：純事實公告（如法說會日期、例行公告）、正負因素相當、對該公司沒有明確影響\n"
            "注意：\n"
            "1. 標題可能為了吸引點閱而誇大或使用反問句，請以內文陳述的事實為準。\n"
            "2. 只評估對指定公司的影響，不是整體市場或其他公司的好壞。\n"
            "3. 「止跌」「反彈」「翻紅」代表回升，是利多。\n"
            "4. 新聞提到競爭對手時，要從指定公司的角度判斷：對手追不上、受到威脅、股價承壓，"
            "代表指定公司地位穩固，是利多；對手追上或搶走訂單，才是利空。\n"
            "reason 用一句繁體中文（80 字以內）寫出影響判斷的具體事實：寫出主詞（誰）和發生什麼事，"
            "盡量包含新聞中的數字（例如營收、成長率、目標價），只能使用新聞中的內容；"
            "不要只寫「屬利多」「屬利空」這類結論。label 只能是 positive、neutral、negative。",
        ),
        (
            "human",
            "指定公司：{stock_name}（{stock_id}）\n\n"
            "標題：{title}\n\n"
            "開頭：{lead}\n\n"
            "提到該公司的句子：\n{mentions}",
        ),
    ]
)

# ---------- v2 ----------
# v1 的問題（台積電 30 天的實測）：
#   ① 利空清單寫「競爭對手威脅」→「英特爾追趕台積電很難，對其地位造成威脅」被判為利空
#   ② 利空清單寫「股價下跌」但沒定義 →「平盤作收、被大盤拖累」被判為利空
#   ③ 沒有先確認新聞在講誰 → 英特爾的困難被當成台積電的困難
# v2：schema 先寫 subject（結構化的思考步驟），修正利多利空的定義，加上判斷步驟


class LLMSentimentV2(BaseModel):
    # 欄位順序就是生成順序：先確認主角，再寫理由，最後下結論
    subject: str = Field(description="這篇新聞主要在講誰的什麼事，30 字以內")
    reason: str = Field(
        description="一句話寫出影響判斷的具體事實（主詞＋發生什麼事），盡量包含新聞中的數字，80 字以內"
    )
    label: Literal["positive", "neutral", "negative"] = Field(
        description="對指定公司而言：positive 利多、neutral 中立、negative 利空"
    )


PROMPT_V2 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是財經新聞分析師，只輸出 JSON。判斷這篇新聞對「指定公司」是利多、利空還是中立。\n"
            "- positive 利多：營收或獲利成長、接獲訂單、法人調升評等或目標價、指定公司股價明顯上漲或止跌回升、"
            "新產品或擴產順利、競爭對手遇到困難或追不上\n"
            "- negative 利空：營收或獲利衰退、訂單或客戶流失、法人調降評等、指定公司股價明顯下跌、訴訟或監管、"
            "競爭對手追上或搶走訂單\n"
            "- neutral 中立：純事實公告（如法說會日期、例行公告）、正負因素相當、對該公司沒有明確影響、"
            "股價平盤或小幅波動、只是隨大盤漲跌\n"
            "判斷步驟：\n"
            "1. 先寫 subject：這篇新聞主要在講誰的什麼事。"
            "如果主要在講競爭對手，要從指定公司的角度判斷：對手遇到困難、追不上，代表指定公司地位穩固，是利多；"
            "對手追上或搶走訂單，才是利空。\n"
            "2. 股價新聞：只有指定公司本身的股價明顯漲跌，且新聞說明了原因，才算利多或利空；"
            "平盤、小幅波動、被大盤或其他股票拖累，是中立。\n"
            "3. 評論或分析文（例如外媒看好、點名買進）可以判為利多或利空，"
            "但 reason 要寫出誰認為什麼、依據是什麼。\n"
            "4. 標題可能為了吸引點閱而誇大或使用反問句，請以內文陳述的事實為準。\n"
            "reason 用一句繁體中文（80 字以內）寫出影響判斷的具體事實：寫出主詞（誰）和發生什麼事，"
            "盡量包含新聞中的數字（例如營收、成長率、目標價），只能使用新聞中的內容；"
            "不要只寫「屬利多」「屬利空」這類結論。label 只能是 positive、neutral、negative。",
        ),
        PROMPT.messages[1],  # human 訊息與 v1 相同
    ]
)

# ---------- v3 ----------
# Gemma 4＋v1 實測（台積電、鴻海 30 天）：正反並陳的分析文，模型會挑出文中「正面的事實」當理由、判為利多，
# 忽略文章的主要論點，例如「鴻海 AI 營收超車，但毛利率僅 6%、股價上不去」被判為利多；
# 地方或生活報導只是順帶提到公司的投資金額，也被判為利多。
# v3：v1 加上第 5、6 點；同時避免矯枉過正：財報等營運消息明確時，不因當天股價下跌改判（第 5 點後半）

V3_RULES = (
    "5. 分析或評論文要依全文的主要論點判斷，不要只挑其中一個正面或負面的事實。"
    "例如「營收創新高，但毛利率偏低、股價落後」的文章，主要論點是後面的疑慮；"
    "正反並陳、沒有明確結論的，判為中立。reason 要寫出文章的主要論點。"
    "但如果新聞的重點是指定公司本身的營運消息（財報、營收、訂單、產能），以營運消息為準，"
    "當天股價或 ADR 的漲跌不改變判斷。\n"
    "6. 地方、生活、人物或歷史報導，只是順帶提到指定公司的投資、規模或過去的事，"
    "對公司沒有新的營運影響，判為中立。\n"
)
PROMPT_V3 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            PROMPT.messages[0].prompt.template.replace(
                "reason 用一句", V3_RULES + "reason 用一句"
            ),
        ),
        PROMPT.messages[1],  # human 訊息與 v1 相同
    ]
)

PROMPTS = {
    "v1": (PROMPT, LLMSentiment),
    "v2": (PROMPT_V2, LLMSentimentV2),
    "v3": (PROMPT_V3, LLMSentiment),
}
_chains = {}  # (模型, prompt 版本) → chain


def get_chain(model=None, version=None):
    """依模型與 prompt 版本取得 chain；第一次呼叫時才建立"""
    key = (model or MODEL, version or PROMPT_VERSION)
    if key not in _chains:
        prompt, schema = PROMPTS[key[1]]
        llm = make_llm(key[0])
        _chains[key] = prompt | llm.with_structured_output(schema, method="json_schema")
    return _chains[key]


def classify(stock_id, stock_name, title, content, retries=1, model=None, version=None):
    hits = find_mentions(stock_id, stock_name, content)
    in_title = stock_name in title or stock_id in title
    inputs = build_inputs(
        stock_id, stock_name, title, content, hits, in_title
    )  # 與相關性判斷相同的輸入
    last_err = None
    for _ in range(retries + 1):
        try:
            return get_chain(model, version).invoke(
                inputs
            )  # 回傳 LLMSentiment 或 LLMSentimentV2
        except Exception as e:
            last_err = e
    raise last_err


if __name__ == "__main__":
    tests = [
        (
            "台積電第三季獲利創新高 法人調升目標價",
            "台積電公布第三季財報，稅後純益年增五成，優於市場預期。\n多家外資調升台積電目標價。",
            "positive",
        ),
        (
            "台積電跌破2400元 外資連續賣超",
            "外資本周大舉賣超台積電，股價跌破2400元，市場擔憂AI需求降溫。",
            "negative",
        ),
        (
            "台積電將於下周四舉行法說會",
            "台積電公告將於10月16日舉行法人說明會，說明第三季營運成果。",
            "neutral",
        ),
        (  # 競爭對手處於劣勢：全量執行時發現 TAIDE 容易判成負面
            "英特爾想撼動台積電地位？外媒直指十分困難",
            "外媒分析指出，英特爾晶圓代工業務要追上台積電十分困難，台積電在先進製程的領先地位短期內難以撼動。",
            "positive",
        ),
        (  # 標題反問、內文正面：nlptown 容易判錯的類型
            "台積電明天免驚？美股盤前止跌翻紅了",
            "科技股近期賣壓沉重。不過美股期指盤前反彈，台積電ADR一度翻紅，為台股明日開盤帶來支撐。",
            "positive",
        ),
    ]
    for title, content, expected in tests:
        r = classify("2330", "台積電", title, content)
        mark = "✓" if r.label == expected else "✗"
        print(
            f"{mark} {r.label:<9}（預期 {expected}）| {title}\n            理由：{r.reason}"
        )
