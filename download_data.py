import urllib.request, bz2, re, os
import html
from src.word2vec.config import URL,RAW_PATH,TOPIC_KEYWORDS, MAX_ARTICLES, USE_TOPIC_KEYWORDS


def to_sentences(text: str):
    raw_sents = re.split(r"[.!?]+", text)
    for s in raw_sents:
        s = s.strip()
        if not s:
            continue
        s = s.lower()
        s = re.sub(r"[^a-z\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        if len(s.split()) >= 3:
            yield s

def remove_nested_braces(text: str) -> str:
    max_iters = 20
    for _ in range(max_iters):
        if '{{' not in text:
            break
        new_text = re.sub(r'\{\{[^{}]*\}\}', ' ', text)
        if new_text == text:
            break
        text = new_text
    text = re.sub(r'\{.*?\}', ' ', text, flags=re.DOTALL)
    return text

def strip_wiki_markup(text: str) -> str:
    text = html.unescape(text)
    text = html.unescape(text)
    text = re.sub(r'<gallery.*?>.*?</gallery>', ' ', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<imagemap.*?>.*?</imagemap>', ' ', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'\[\[category:.*?\]\]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'<ref[^>]*>.*?</ref>', ' ', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<ref[^>]*/>', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\{\|.*?\|\}', ' ', text, flags=re.DOTALL)
    text = re.sub(r'\[\[(File|Image):.*?\]\]', ' ', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]*)\]\]', r'\1', text)
    text = remove_nested_braces(text)
    text = re.sub(r'={2,}[^=]+={2,}', ' ', text)
    text = re.sub(r'__[A-Z]+__', ' ', text)
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'www\.\S+', ' ', text)
    text = re.sub(r'<.*?>', ' ', text, flags=re.DOTALL)
    text = re.sub(r'\b(file|image)\s*:\s*\S+', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(thumb|px|right|left|center|upright|frame|wikitext|rect)\b', ' ', text)
    text = re.sub(r'\btext\s+x\s+wiki\b', ' ', text, flags=re.IGNORECASE)
    return text



def matches_topic(title: str, keywords: list) -> bool:
    title_lower = title.lower()
    return any(kw in title_lower for kw in keywords)


def stream_articles(path):
    buf = ""
    in_text = False
    current_title = ""
    with bz2.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            buf += line

            title_match = re.search(r"<title>(.*?)</title>", buf)
            if title_match:
                current_title = title_match.group(1).strip()
                buf = buf[title_match.end():]

            if not in_text:
                start = buf.find("<text")
                if start != -1:
                    tag_end = buf.find(">", start)
                    if tag_end != -1:
                        in_text = True
                        buf = buf[tag_end + 1:]

            if in_text:
                end = buf.find("</text>")
                if end != -1:
                    content = buf[:end].strip()
                    buf = buf[end + 7:]
                    in_text = False
                    if content:
                        yield current_title, content

            if len(buf) > 1_000_000:
                buf = ""


os.makedirs("data/raw", exist_ok=True)
print("Download started")
urllib.request.urlretrieve(URL, "/tmp/simplewiki.xml.bz2")

count = 0
with open(RAW_PATH, "w", encoding="utf-8") as out:
    for title, text in stream_articles("/tmp/simplewiki.xml.bz2"):
        if text.strip().lower().startswith("#redirect"):   # skip redirect pages
            continue
        if USE_TOPIC_KEYWORDS and not matches_topic(title, TOPIC_KEYWORDS):
            continue                          # skip off-topic articles
        text = strip_wiki_markup(text)
        for sent in to_sentences(text):
            out.write(sent + "\n")
        count += 1
        print(f"wrote acticle {count}")
        if count >= MAX_ARTICLES:
            break

print(f"Done. Wrote {count} articles to {RAW_PATH}")
