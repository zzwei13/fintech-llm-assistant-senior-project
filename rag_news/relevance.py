"""
TAIDE 相關性判斷（ingest 階段 ②）

判斷一篇新聞的「主角」是否為目標公司，三級分類：
    main       新聞主要在講這家公司
    mention    有提到，但主角是其他公司或整體產業
    unrelated  沒有實質提到這家公司

做法：規則＋是非題
  1. 規則：標題與內文都沒有公司名稱或代號 → unrelated，不呼叫模型
  2. 規則：標題以「公司名稱:」開頭（公司公告）→ main
  3. 規則：標題沒有公司名稱或代號 → mention，不呼叫模型（主角必須出現在標題）
  4. TAIDE 只回答一個是非題：「這家公司是不是新聞的主角？」 → 是：main／否：mention
  只有 main 會計入情緒分數，所以寧可把 main 判嚴一點（main 的精確率優先）。
  模型只做二元判斷，分類由程式對應，避免「理由說 mention、標籤卻輸出 main」的矛盾。

相較舊版 together_filter.py 的改良：
  ① 判斷標準直接寫入目標公司名稱與代號（舊版是「與股市無關」）
  ② 單一任務：只判斷相關性，不混入獲利預測
  ③ JSON schema 約束輸出、temperature 0（舊版用字串包含「好」解析、temperature 0.7）
  ④ 輸入「標題＋開頭＋提到該公司的句子」，而不是 sumy 摘要
  ⑤ 只標記不刪除；⑥ ingest 時每篇只判斷一次

單獨測試：
    python relevance.py
"""

import os
import re
from typing import Literal

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

MODEL = os.getenv("OLLAMA_MODEL", "gemma4:12b")
# prompt 版本：v1 原版；v2 把「盤勢、外資買賣超排行中只是成分股」的新聞判為非主角（compare_models.py 比較用）
PROMPT_VERSION = os.getenv("RELEVANCE_PROMPT", "v1")
LEAD_CHARS = 300  # 開頭最多取幾個字
MAX_MENTIONS = 5  # 最多附上幾句提到公司的句子
MENTION_CHARS = 120  # 每句最多幾個字
NUM_PREDICT = 512  # 最多生成幾個 token：避免模型停不下來
LLM_TIMEOUT = 180  # 單次請求最多等幾秒：超過就視為失敗，不會整個程式卡住
THINKING_MODELS = (
    "gemma4",
    "qwen3",
)  # 有思考模式的模型，關掉思考：分類任務不需要，而且會大幅拖慢速度


def make_llm(model, num_ctx=4096, num_predict=NUM_PREDICT):
    """
    建立 ChatOllama：temperature 0、限制生成長度、設定逾時；有思考模式的模型關掉思考。
    相關性、情緒判斷共用（llm_sentiment.py 也從這裡匯入）。
    """
    kwargs = dict(
        model=model,
        temperature=0,
        num_ctx=num_ctx,
        num_predict=num_predict,
        client_kwargs={"timeout": LLM_TIMEOUT},
    )
    if model.startswith(THINKING_MODELS):
        try:
            return ChatOllama(reasoning=False, **kwargs)
        except Exception:  # 舊版 langchain-ollama 沒有 reasoning 參數
            pass
    return ChatOllama(**kwargs)


# schema -> 模型回傳的 JSON schema
class LLMOutput(BaseModel):
    # 先寫理由，再下結論
    reason: str = Field(description="一句話說明判斷依據，60 字以內")
    is_main: bool = Field(description="指定公司是否為這篇新聞的主角")


# classify() 回傳給 ingest 的結果，relevance 對應資料庫 ENUM('main','mention','unrelated')
class Relevance(BaseModel):
    reason: str
    relevance: Literal["main", "mention", "unrelated"]
    by_rule: bool = False  # True：由規則判斷，沒有呼叫模型


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是財經新聞分類器，只輸出 JSON。判斷指定公司是不是這篇新聞的主角。\n"
            "- 是主角：新聞主要在講這家公司，例如它的營收、獲利、股價、法說會、訂單、產品、人事\n"
            "- 不是主角：新聞主要在講其他公司、整體產業、大盤或一般理財話題，這家公司只是被舉例或順帶提到\n"
            "- 判斷方式：把這家公司從新聞中拿掉，新聞還能成立，就不是主角。"
            "同時討論多家公司、整個產業或總體經濟的分析文章，通常不是主角。\n"
            "reason 用一句繁體中文（60 字以內）說明依據；is_main 為 true 代表是主角，false 代表不是。",
        ),
        (
            "human",
            "指定公司：{stock_name}（{stock_id}）\n\n"
            "標題：{title}\n\n"
            "開頭：{lead}\n\n"
            "標題是否提到該公司：{in_title}；內文提到該公司的句數：{n_hits}\n"
            "提到該公司的句子：\n{mentions}",
        ),
    ]
)

PROMPT_V2 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是財經新聞分類器，只輸出 JSON。判斷指定公司是不是這篇新聞的主角。\n"
            "- 是主角：新聞主要在講這家公司本身，例如它的營收、獲利、法說會、訂單、產品、人事、評等或目標價，"
            "或專門分析這家公司的股價與前景\n"
            "- 不是主角：新聞主要在講其他公司、整體產業、大盤或一般理財話題，這家公司只是被舉例或順帶提到\n"
            "- 盤勢新聞不是主角：新聞主要在描述大盤漲跌、類股表現、外資或法人買賣超排行，"
            "這家公司只是被列出的其中一檔權值股或成分股，即使標題提到它也一樣\n"
            "- 判斷方式：把這家公司從新聞中拿掉，新聞還能成立，就不是主角。"
            "同時討論多家公司、整個產業或總體經濟的分析文章，通常不是主角。\n"
            "reason 用一句繁體中文（60 字以內）說明依據；is_main 為 true 代表是主角，false 代表不是。",
        ),
        PROMPT.messages[1],  # human 訊息與 v1 相同
    ]
)

PROMPTS = {"v1": PROMPT, "v2": PROMPT_V2}
_chains = {}  # (模型, prompt 版本) → chain


def get_chain(model=None, version=None):
    """依模型與 prompt 版本取得 chain；第一次呼叫時才建立"""
    key = (model or MODEL, version or PROMPT_VERSION)
    if key not in _chains:
        llm = make_llm(key[0])
        _chains[key] = PROMPTS[key[1]] | llm.with_structured_output(
            LLMOutput, method="json_schema"
        )  # 結果轉成 LLMOutput 物件
    return _chains[key]


def find_mentions(stock_id, stock_name, content):
    """所有提到公司名稱或代號的句子"""
    sentences = re.split(r"(?<=[。！？])", content.replace("\n", ""))
    keys = (stock_name, stock_id)
    return [s.strip() for s in sentences if any(k in s for k in keys)]


def build_inputs(stock_id, stock_name, title, content, hits, in_title):
    """組出 prompt 需要的輸入"""
    # 開頭兩段
    paragraphs = [p for p in content.split("\n") if p.strip()]
    lead = "".join(paragraphs[:2])[:LEAD_CHARS]
    # 提到公司的句子（最多 5 句、每句最多 120 字）
    mentions = (
        "\n".join(f"- {s[:MENTION_CHARS]}" for s in hits[:MAX_MENTIONS])
        or "內文中沒有出現公司名稱或代號"
    )

    return {
        "stock_id": stock_id,
        "stock_name": stock_name,
        "title": title,
        "lead": lead,
        "mentions": mentions,
        "in_title": "是" if in_title else "否",
        "n_hits": len(hits),
    }


def classify(stock_id, stock_name, title, content, retries=1, model=None, version=None):
    hits = find_mentions(stock_id, stock_name, content)
    in_title = stock_name in title or stock_id in title

    # 1. 規則：標題與內文都沒提到 → unrelated，不呼叫模型
    if not hits and not in_title:
        return Relevance(
            reason="標題與內文皆未出現公司名稱或代號",
            relevance="unrelated",
            by_rule=True,
        )

    # 2. 規則：標題以「公司名稱:」開頭 → main（公開資訊觀測站重大訊息、公司本身的聲明）
    if re.match(rf"{re.escape(stock_name)}[:：]", title):
        return Relevance(
            reason="公司公告或聲明（標題以公司名稱開頭）",
            relevance="main",
            by_rule=True,
        )

    # 3. 規則：標題沒提到 → mention，不呼叫模型（主角必須出現在標題）
    if not in_title:
        return Relevance(
            reason=f"標題未出現公司，內文 {len(hits)} 句提到",
            relevance="mention",
            by_rule=True,
        )

    # 4. 模型：只判斷是不是主角
    inputs = build_inputs(stock_id, stock_name, title, content, hits, in_title)
    last_err = None
    for _ in range(retries + 1):
        try:
            out = get_chain(model, version).invoke(inputs)  # 回傳 LLMOutput
            return Relevance(
                reason=out.reason, relevance="main" if out.is_main else "mention"
            )
        except Exception as e:  # 格式錯誤、連線失敗等
            last_err = e
    raise last_err  # 拋出錯誤


if __name__ == "__main__":
    tests = [
        (
            "2330",
            "台積電",
            "台積電跌破2400元 內資報告抱屈喊「評價偏低」",
            "台股權值王台積電（2330）自今年高點2590元一路下殺，上周四更跌破2400元大關。\n元大投顧指出，台積電第三季N7以下製程比重來到79%。",
            "main",
        ),
        (
            "2330",
            "台積電",
            "欣興、景碩與南電股價香噴 ABF載板漲風拉升目標價",
            "台灣IC載板三強欣興、景碩與南電正在擴大生產基地。\n基板是半導體供應鏈的關鍵環節，它使台積電等公司能夠將晶片封裝在單一模組中。",
            "mention",
        ),
        (
            "2330",
            "台積電",
            "年輕人最該投資什麼？網友點名「這些事」別等老了才後悔",
            "網友在社群平台發文詢問年輕人最該投資什麼。\n有網友點名0050、00631L及台積電（2330）等標的，也有人認為健康最重要。",
            "mention",
        ),
        (
            "2330",
            "台積電",
            "歷史首次！中國車廠在歐洲季度銷量超過日本",
            "在一個完整季度內，中國汽車製造商在歐洲的銷量首次超過日本競爭對手。\n9月中國品牌在歐洲的市佔率升至14.5%。",
            "unrelated",
        ),
    ]
    for sid, name, title, content, expected in tests:
        r = classify(sid, name, title, content)
        mark = "✓" if r.relevance == expected else "✗"
        src = "規則" if r.by_rule else "模型"
        print(
            f"{mark} {r.relevance:<10}（預期 {expected}，{src}）| {title[:24]}\n           理由：{r.reason}"
        )
