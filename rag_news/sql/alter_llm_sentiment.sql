-- 新增 TAIDE 情緒欄位（階段 3b）
-- 執行：mysql -u root -p stock_news < alter_llm_sentiment.sql
ALTER TABLE news
  ADD COLUMN llm_sentiment_label  ENUM('negative', 'neutral', 'positive') NULL AFTER star_probs,
  ADD COLUMN llm_sentiment_reason VARCHAR(255) NULL AFTER llm_sentiment_label;