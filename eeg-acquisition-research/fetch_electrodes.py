#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量查询 OpenAlex + PubMed 获取脑电电极/去噪文献元数据。存 JSON，打印紧凑摘要。"""
import json, os, time, urllib.parse, urllib.request

OUT = os.path.expanduser("~/BCI/eeg-acquisition-research/raw")
os.makedirs(OUT, exist_ok=True)

UA = {"User-Agent": "Mozilla/5.0 (research; contact: researcher@example.com)"}

def get(url, timeout=45):
    req = urllib.request.Request(url, headers=UA)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            if attempt == 2:
                return None
            time.sleep(2 + attempt * 2)

# ---------------- OpenAlex ----------------
OA_QUERIES = [
    "semi-dry electrode",
    "dry electrode EEG",
    "graphene electrode EEG",
    "e-tattoo EEG electrode",
    "epidermal electronics EEG",
    "ear-EEG",
    "in-ear EEG",
    "active electrode EEG",
    "high-density EEG electrode",
    "flexible electrode EEG",
    "carbon nanotube EEG electrode",
    "hydrogel electrode EEG",
    "conductive polymer electrode EEG PEDOT",
    "microneedle electrode EEG",
    "electrogel electrode",
    "saline sponge electrode EEG",
    "EEG artifact removal deep learning",
    "Artifact Subspace Reconstruction EEG",
    "EEG motion artifact removal",
    "independent component analysis EEG artifact",
    "electrode-skin impedance EEG",
    "EEG signal quality assessment",
]

oa_data = {}
for q in OA_QUERIES:
    url = ("https://api.openalex.org/works?search=" + urllib.parse.quote(q)
           + "&per-page=12&sort=relevance_score:desc"
           + "&select=id,title,publication_year,doi,primary_location,cited_by_count,authorships")
    txt = get(url)
    hits = []
    if txt:
        try:
            j = json.loads(txt)
            for w in j.get("results", []):
                src = None
                pl = w.get("primary_location") or {}
                if pl.get("source"):
                    src = pl["source"].get("display_name")
                authors = "; ".join(a.get("author", {}).get("display_name", "?") for a in (w.get("authorships") or [])[:3])
                hits.append({
                    "title": (w.get("title") or "").strip(),
                    "year": w.get("publication_year"),
                    "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
                    "venue": src,
                    "cites": w.get("cited_by_count"),
                    "authors": authors,
                })
        except Exception as e:
            print("OA parse err", q, e)
    oa_data[q] = hits
    print(f"[OA] {q}: {len(hits)} hits")
    time.sleep(0.4)

with open(os.path.join(OUT, "openalex_electrodes.json"), "w") as f:
    json.dump(oa_data, f, ensure_ascii=False, indent=1)
print("saved openalex_electrodes.json")
