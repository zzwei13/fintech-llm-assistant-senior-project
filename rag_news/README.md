# 新聞情緒分析＋RAG 個股新聞摘要（rag_news）

大學畢業專題「漲LLa!Ma?」新聞分析的新版。用媒體報導對個股的傾向作為投資人情緒的參考指標，並用 RAG 整理「指標由哪些報導構成」。全部在本機執行。

> 定位：整理近期新聞作為參考，**不預測股價、不提供投資建議**。

## 和舊版的差異

| | 舊版（2024） | 新版 |
|---|---|---|
| 情緒 | BERT（nlptown）判斷整篇語氣 | LLM 判斷「對指定公司」的影響，附理由 |
| 相關性 | 與股市是否有關 | 這家公司是不是新聞的主角（主角／提及／無關） |
| 摘要 | 30 天新聞摘要交給 LLM，摘要和新聞沒有對應關係 | RAG：每個重點對應到它引用的新聞，並標出對情緒指標的貢獻 |
| 執行 | 雲端 API，每次查詢重算 | 本機（Ollama），資料進來時算一次＋快取 |

## 架構

```
資料準備（ingest.py，每個階段可單獨重跑、中斷續跑）
  ① crawl          自由時報＋鉅亨網 → MySQL（一篇一筆）
  ② relevance      規則＋LLM 是非題 → main／mention／unrelated
  ③ sentiment      nlptown（baseline，只用來比較）
  ③b llm-sentiment LLM 目標導向情緒＋理由（只限 main）
  ④ index          切塊 → bge-m3 → Chroma

查詢（app.py → chain.py）
  1. 情緒指標與每篇貢獻（trend.py）    s ∈ {+1,0,−1}、w = 0.5^(天數/7)、c_i = s_i·w_i ／ Σw
  2. 報導焦點（events.py）             以種子檢索分群：bge-m3（標題＋理由）、相似度 ≥ 0.65、相隔 ≤ 5 天
  3. 焦點內檢索證據片段（vectorstore.py）
  4. 重點 = 代表新聞的判斷理由（數字核對；不通過改用原文句子）
  5. 焦點命名：LLM 呼叫 1 次；摘要由程式組成
```

## 環境

- Python（conda 環境 `rag-news`）：`pip install -r requirements.txt`
- MySQL：`sql/schema.sql` 建表，`sql/import_stocks.sql` 匯入股票清單（`data/stock.csv`）；舊資料庫升級用 `sql/alter_llm_sentiment.sql`
- Ollama：`ollama pull gemma4:12b`、`ollama pull bge-m3`
- 爬蟲需要 Google Chrome（Selenium）

`.env`（不放進 git）：

| 變數 | 說明 | 預設 |
|---|---|---|
| `DB_HOST`、`DB_PORT`、`DB_USER`、`DB_PASSWORD`、`DB_NAME` | MySQL 連線 | localhost／3306／root／（空）／stock_news |
| `OLLAMA_MODEL` | 相關性、情緒、焦點命名用的 LLM | 建議設為 `gemma4:12b` |
| `EMBEDDING_MODEL` | 向量模型 | bge-m3 |
| `RELEVANCE_PROMPT`、`SENTIMENT_PROMPT` | prompt 版本 | v1、v3 |

## 使用

```bash
# 資料準備
python ingest.py crawl --stock 2330 2317 2002 --days 30
python ingest.py relevance
python ingest.py llm-sentiment
python ingest.py index
python ingest.py status

# 日常更新（一鍵增量：爬新新聞 → 判斷 → 索引 → 預先產生 RAG 快取，頁面顯示更新時間）
python ingest.py update                      # 所有已有新聞的股票
python ingest.py update --stock 2330 2454    # 指定股票；新股票會抓近 30 天

# 換模型或改 prompt 後全部重新判斷（會先備份到 backup/，並同步 Chroma）
python ingest.py relabel --stock 2330 2317 2002
python ingest.py relabel --stock 2330 2317 2002 --sentiment-only   # 只重跑情緒
python ingest.py relabel --stock 2330 2317 2002 --resume           # 中斷後接著跑

# RAG 分析（結果快取在 cache/；資料或參數改變時自動重算）
python chain.py --stock 2330 --end 2026-10-26

# 頁面
python app.py   # http://127.0.0.1:5001/?stock=2330&end=2026-10-26
```

## 資料更新

LLM 判斷每篇需要數秒，不適合在使用者點擊時才爬新聞和判斷；所以資料在背景用 `ingest.py update` 更新，頁面只讀已算好的結果與快取，並顯示資料更新時間。`update` 每個階段只處理新新聞，重複執行或中斷後重跑都是安全的。

每天自動更新（macOS／Linux，電腦需開著），例如每天 07:30：

```
30 7 * * * cd /path/to/rag_news && /path/to/conda/envs/rag-news/bin/python ingest.py update >> update.log 2>&1
```

## 評估

沒有人工標註資料，所以用不需要標註的方法評估；結果都輸出到 `compare/`。

| 程式 | 評估什麼 |
|---|---|
| `eval_sentiment.py` | 情緒判斷的 baseline 比較：nlptown vs LLM，18 題一般測試（競爭對手、否定、轉折、標題誇大、多家公司、純事實公告，正中負各 6 題） |
| `compare_models.py` | 模型與 prompt 版本比較（回歸測試＋真實新聞並排輸出） |
| `eval_rag.py` | RAG 檢索與生成：涵蓋率、內聚度、重複度、相關片段比例、證據支持率、數字核對、語意支持度、名稱規則，以及 LLM 評審（Qwen 3.5，不用產生結果的模型） |
| `eval_ablation.py` | 消融實驗：拿掉分群、日期限制、medoid、片段排序、原文補強，3 檔 × 3 個結束日 |

主要結果（2026-09-27～10-26）：

- 情緒：nlptown 5/18（一題都沒判成正面）、Gemma 4＋v3 18/18
- RAG：數字核對 100%、證據支持率 100%、LLM 評審判方向相反 0%
- 消融：拿掉分群，涵蓋率 64% → 31%、出現重複焦點；拿掉原文補強，數字核對 100% → 89%

限制：測試題少且比真實新聞單純，不能當作準確率；LLM 評審本身不穩定，只用來篩出需要人工檢查的項目；同一事件的不同報導角度可能被分成兩個焦點。

## 檔案

| 檔案 | 說明 |
|---|---|
| `crawler_ltn.py`、`crawler_cnyes.py` | 爬蟲 |
| `db.py` | MySQL 存取（PyMySQL，不使用 ORM） |
| `ingest.py` | 資料管線 |
| `relevance.py`、`llm_sentiment.py` | 相關性、情緒判斷（prompt 版本在檔案內） |
| `sentiment.py` | nlptown baseline |
| `compare_input.py` | nlptown 輸入比較（標題＋開頭 vs 只有開頭） |
| `vectorstore.py` | 切塊、向量化、檢索（Chroma） |
| `trend.py` | 情緒指標、貢獻度、趨勢圖 |
| `events.py` | 報導焦點分群 |
| `chain.py` | RAG：證據、重點、命名、摘要、快取 |
| `app.py`、`templates/index.html` | Flask 頁面 |
| `test_cases.py` | 回歸測試與一般測試題目 |
| `sql/`、`data/stock.csv` | 資料表定義、股票清單 |
| `eval_*.py`、`compare_models.py` | 評估 |

不放進 git：`.env`（範本見 `.env.example`）、`chroma_db/`、`cache/`、`compare/`、`backup/`、`scratch/`（含新聞內容與判斷結果，可由程式重新產生）。