# scripts/m1_compare_vec4.py
#!/usr/bin/env python3
import sys, csv

def load(path):
    rows=[]
    with open(path, newline="") as f:
        r=csv.DictReader(f)
        for row in r: rows.append(row)
    return rows

def pick(rows, engine="autosage"):
    byF={}
    for r in rows:
        if r["engine"]!=engine: continue
        F=int(r["F"]); byF[F]=float(r["median_ms"])
    return byF

if __name__=="__main__":
    if len(sys.argv)!=3:
        print("usage: m1_compare_vec4.py vec4_on.csv vec4_off.csv"); sys.exit(1)
    on = pick(load(sys.argv[1]))
    off= pick(load(sys.argv[2]))
    Fs = sorted(set(on.keys()) & set(off.keys()))
    print("F   autosage(ms) vec4(ms)  uplift(%)")
    for F in Fs:
        t_on, t_off = on[F], off[F]
        upl = (t_off/t_on - 1.0)*100.0
        print(f"{F:<4d} {t_off:>10.3f} {t_on:>9.3f}  {upl:>8.1f}")
