-- 新聞情緒分析：MySQL 資料庫與資料表
-- 執行：mysql -u root -p < rag_news/schema.sql

CREATE DATABASE IF NOT EXISTS stock_news
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE stock_news;

CREATE TABLE IF NOT EXISTS stocks (
  stock_id   VARCHAR(10) PRIMARY KEY,
  stock_name VARCHAR(50) NOT NULL
);

CREATE TABLE IF NOT EXISTS news (
  id               INT AUTO_INCREMENT PRIMARY KEY,
  stock_id         VARCHAR(10)  NOT NULL,
  date             DATE         NOT NULL,
  published_at     DATETIME     NOT NULL,
  source           VARCHAR(20)  NOT NULL,
  title            VARCHAR(255) NOT NULL,
  url              VARCHAR(512) NOT NULL,                       -- 同一篇可屬於多檔股票
  content          MEDIUMTEXT   NOT NULL,
  relevance        ENUM('main', 'mention', 'unrelated') NULL,  -- TAIDE 相關性判斷
  relevance_reason VARCHAR(255) NULL,                         -- 判斷理由
  sentiment        FLOAT        NULL,                         -- 期望星等換算的 -1～+1
  sentiment_label  ENUM('negative', 'neutral', 'positive') NULL,
  star_probs       JSON         NULL,                         -- [p1, p2, p3, p4, p5]
  llm_sentiment_label  ENUM('negative', 'neutral', 'positive') NULL,  -- TAIDE 對該公司的利多／利空／中立（只限 main）
  llm_sentiment_reason VARCHAR(255) NULL,                     -- TAIDE 判斷理由
  created_at       TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (stock_id) REFERENCES stocks (stock_id),
  UNIQUE KEY uq_stock_url (stock_id, url),                      -- 同一檔股票內網址不重複
  INDEX idx_stock_date (stock_id, date)
);