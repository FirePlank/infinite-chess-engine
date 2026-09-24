#!/usr/bin/env python3
"""Concatenate AEVDAT01 record files with identical headers: merge_data.py out in1 in2 ..."""
import struct, sys
HEADER = struct.Struct("<8sIIQIQ")
out, ins = sys.argv[1], sys.argv[2:]
total, header0 = 0, None
with open(out, "wb") as fo:
    fo.write(b"\0" * HEADER.size)
    for p in ins:
        with open(p, "rb") as fi:
            h = HEADER.unpack(fi.read(HEADER.size))
            if header0 is None:
                header0 = h
            assert h[:5] == header0[:5], (p, h[:5], header0[:5])
            total += h[5]
            while chunk := fi.read(1 << 26):
                fo.write(chunk)
    fo.seek(0)
    fo.write(HEADER.pack(*header0[:5], total))
print(f"{out}: {total:,} records from {len(ins)} files")
