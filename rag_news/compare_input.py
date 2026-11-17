"""
比較 nlptown 的兩種輸入：「標題 + 開頭兩段」vs「只有開頭兩段」
並印出開頭內容，方便依內文（而不是標題）判斷正負面。

    python compare_input.py            # 已算過情緒的新聞
    python compare_input.py --limit 20
"""

import argparse

import db
from sentiment import analyze, build_text


def lead_only(content):
    paragraphs = [p for p in content.split("\n") if p.strip()]
    return "".join(paragraphs[:2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    with db.get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, relevance, title, content FROM news "
            "WHERE sentiment IS NOT NULL ORDER BY published_at DESC LIMIT %s",
            (args.limit,),
        )
        rows = cur.fetchall()

    with_title = analyze([build_text(r["title"], r["content"]) for r in rows])
    no_title = analyze([lead_only(r["content"]) for r in rows])

    for r, a, b in zip(rows, with_title, no_title):
        diff = b["sentiment"] - a["sentiment"]
        print(f"[{r['relevance']}] {r['title'][:40]}")
        print(
            f"  標題+開頭 {a['sentiment']:+.2f} {a['sentiment_label']:<9}"
            f"只有開頭 {b['sentiment']:+.2f} {b['sentiment_label']:<9}差 {diff:+.2f}"
        )
        print(f"  開頭：{lead_only(r['content'])[:120]}\n")

    changed = sum(
        a["sentiment_label"] != b["sentiment_label"]
        for a, b in zip(with_title, no_title)
    )
    avg_a = sum(a["sentiment"] for a in with_title) / len(rows)
    avg_b = sum(b["sentiment"] for b in no_title) / len(rows)
    print(
        f"共 {len(rows)} 篇，標籤改變 {changed} 篇；平均分數 標題+開頭 {avg_a:+.2f}、只有開頭 {avg_b:+.2f}"
    )


if __name__ == "__main__":
    main()
