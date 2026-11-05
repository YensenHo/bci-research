#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PubMed E-utilities + arXiv export 批量查询。存 JSON，打印摘要。"""
import json, os, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET

OUT = os.path.expanduser("~/BCI/eeg-acquisition-research/raw")
os.makedirs(OUT, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (research; contact: researcher@example.com)"}

def get(url, timeout=50):
    req = urllib.request.Request(url, headers=UA)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except Exception:
            if attempt == 2:
                return None
            time.sleep(2 + attempt * 2)

# ---------------- PubMed ----------------
PM_QUERIES = [
    "semi-dry EEG electrode",
    "dry EEG electrode",
    "graphene electrode electroencephalography",
    "ear-EEG",
    "high-density EEG electrode",
    "electrode-skin impedance EEG",
    "EEG artifact removal deep learning",
    "Artifact Subspace Reconstruction",
    "motion artifact EEG removal",
    "active electrode EEG amplifier",
]

def esearch(term, retmax=8):
    url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json&retmax="
           + str(retmax) + "&term=" + urllib.parse.quote(term))
    txt = get(url)
    if not txt:
        return []
    try:
        return json.loads(txt)["esearchresult"]["idlist"]
    except Exception:
        return []

def esummary(pmids):
    if not pmids:
        return {}
    url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&retmode=json&id="
           + ",".join(pmids))
    txt = get(url)
    if not txt:
        return {}
    try:
        j = json.loads(txt)
        out = {}
        for pid, rec in j.get("result", {}).items():
            if pid == "uids":
                continue
            out[pid] = {
                "title": rec.get("title", ""),
                "year": (rec.get("pubdate", "") or "")[:4],
                "journal": rec.get("fulljournalname", "") or rec.get("source", ""),
                "authors": "; ".join(a.get("name", "?") for a in rec.get("authors", [])[:3]),
            }
        return out
    except Exception:
        return {}

pm_data = {}
for q in PM_QUERIES:
    ids = esearch(q)
    meta = esummary(ids)
    pm_data[q] = [{"pmid": pid, **meta.get(pid, {})} for pid in ids]
    print(f"[PM] {q}: {len(ids)} ids")
    time.sleep(0.35)

with open(os.path.join(OUT, "pubmed_electrodes.json"), "w") as f:
    json.dump(pm_data, f, ensure_ascii=False, indent=1)
print("saved pubmed_electrodes.json")
