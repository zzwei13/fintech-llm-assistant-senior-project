-- 從 rag_news/data/stock.csv（複製自舊版 StockCrawlingCode/per_roa_crawling/stock.csv）
-- 匯入上市股票清單到 stocks 表
-- 須在 repo 根目錄執行（CSV 路徑為相對路徑）：
--   mysql --local-infile=1 -u root -p < rag_news/import_stocks.sql
-- 重複執行安全：已存在的代號只更新名稱

USE stock_news;

-- 1. 先載入暫存表（連線結束後自動刪除）
CREATE TEMPORARY TABLE stocks_import (
  stock_id   VARCHAR(10),
  stock_name VARCHAR(50)
);

LOAD DATA LOCAL INFILE 'rag_news/data/stock.csv'
INTO TABLE stocks_import
CHARACTER SET utf8mb4
FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(stock_id, stock_name);

-- 2. 寫入正式表：只取 4 碼代號，去除前後空白與換行
INSERT INTO stocks (stock_id, stock_name)
SELECT * FROM (
  SELECT TRIM(stock_id) AS sid, TRIM(TRAILING '\r' FROM TRIM(stock_name)) AS sname
  FROM stocks_import
  WHERE TRIM(stock_id) REGEXP '^[0-9]{4}$'
) AS src
ON DUPLICATE KEY UPDATE stock_name = src.sname;

-- 3. 確認結果
SELECT COUNT(*) AS total_stocks FROM stocks;
SELECT * FROM stocks WHERE stock_id IN ('2330', '2002', '3443');