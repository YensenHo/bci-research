#!/usr/bin/env python3
"""dump header fields of an EDF"""
import sys
import pyedflib

p = sys.argv[1]
f = pyedflib.EdfReader(p)
for attr in ["getPatientName", "getPatientCode", "getPatientSex", "getPatientBirthdate",
             "getRecordingAdditional", "getTechnician", "getEquipment",
             "getStartdatetime", "getAdmincode", "getGender", "getSubtype"]:
    try:
        v = getattr(f, attr)()
        print(f"{attr}: {v!r}")
    except Exception as e:
        print(f"{attr}: ERR {e}")
f.close()
print("--- raw header (first 256 bytes) ---")
with open(p, "rb") as fh:
    hdr = fh.read(256)
print(hdr.decode("latin1").replace("\x00", "·"))
