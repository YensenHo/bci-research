#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""arXiv export API 查询（HTTPS，5s 间隔）。存 JSON。"""
import json, os, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET

OUT = os.path.expanduser("~/BCI/eeg-acquisition-research/raw")
os.makedirs(OUT, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (research)"}
NS = {"a": "http://www.w3.org/2005/Atom"}

def get(url, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except Exception:
            if attempt == 2:
                return None
            time.sleep(4)

ARXIV_QUERIES = [
    'all:"semi-dry electrode"',
    'all:"dry electrode" AND all:EEG',
    'all:"graphene" AND all:EEG AND all:electrode',
    'all:"e-tattoo" AND all:electrode',
    'all:"ear-EEG"',
    'all:"in-ear EEG"',
    'all:"active electrode" AND all:EEG',
    'all:"high-density EEG"',
    'all:"EEG artifact removal" AND all:"deep learning"',
    'all:"Artifact Subspace Reconstruction"',
    'all:"electrode-skin impedance" AND all:EEG',
    'all:"motion artifact" AND all:EEG',
    'all:"conductive polymer" AND all:EEG AND all:electrode',
]

def parse(xml_txt):
    out = []
    try:
        root = ET.fromstring(xml_txt)
        for e in root.findall("a:entry", NS):
            title = " ".join((e.find("a:title", NS).text or "").split())
            aid = (e.find("a:id", NS).text or "").split("/abs/")[-1]
            pub = (e.find("a:published", NS).text or "")[:10]
            auth = "; ".join(a.find("a:name", NS).text for a in e.findall("a:author", NS)[:3])
            summ = " ".join((e.find("a:summary", NS).text or "").split())[:300]
            out.append({"id": aid, "title": title, "published": pub, "authors": auth, "abstract": summ})
    except Exception as e:
        print("parse err", e)
    return out

arx_data = {}
for q in ARXIV_QUERIES:
    url = "https://export.arxiv.org/api/query?search_query=" + urllib.parse.quote(q) + "&max_results=8&sortBy=relevance"
    txt = get(url)
    hits = parse(txt) if txt else []
    arx_data[q] = hits
    print(f"[ARX] {q[:50]}: {len(hits)} hits")
    time.sleep(5)

with open(os.path.join(OUT, "arxiv_electrodes.json"), "w") as f:
    json.dump(arx_data, f, ensure_ascii=False, indent=1)
print("saved arxiv_electrodes.json")
