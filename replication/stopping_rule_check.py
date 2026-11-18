"""Free check (2026-11-17): replay loss-p2 vs F1-p3 stopping on saved runs.
Uses each run's per-task whole-val losses (recorded since Nov 10) as the true
averaged val loss, vs the Keras val_loss (last val batch only).
Usage: download result JSONs with `modal volume get multitask-outputs /<dir> vol/`
then `python stopping_rule_check.py vol`."""
import json,glob,statistics as st,sys
def replay(vals, p, better):
    best_i=0; since=0
    for i in range(1,len(vals)):
        if better(vals[i],vals[best_i]): best_i=i; since=0
        else:
            since+=1
            if since>=p: return best_i, False
    return best_i, True   # censored: never stalled within records
lo=lambda a,b:a<b; hi=lambda a,b:a>b
rows=[]
for f in sorted(glob.glob(sys.argv[1]+"/*/*.json")):
    d,name=f.split('/')[1][19:],f.split('/')[2][:-5]
    j=json.load(open(f)); R=j.get("epoch_scores")
    if not isinstance(R,list) or not R: continue
    if not any(isinstance(v,dict) and "loss" in v for v in R[0]["val"].values()): continue
    if j.get("self_distill") or j.get("stage2") or name.startswith("single_"): continue
    epe=(j.get("args") or {}).get("evals_per_epoch") or 1
    true=[st.mean(v["loss"] for v in r["val"].values()) for r in R]
    kl=[r["val_loss"] for r in R]
    vf=[r["val_f1"] for r in R]; tf=[100*r["test_f1"] for r in R]
    ep=[r["epoch"] for r in R]
    pertask_min=[ep[min(range(len(R)),key=lambda i:R[i]["val"][k]["loss"])] for k in R[0]["val"]]
    lt,ct=replay(true,2*epe if epe>1 else 2,lo); lk,ck=replay(kl,2*epe if epe>1 else 2,lo); fb,cf=replay(vf,3*epe if epe>1 else 3,hi)
    rows.append(dict(run=f"{d}/{name}",epe=epe,n=len(R),
        true_min=ep[true.index(min(true))], kl_min=ep[kl.index(min(kl))], f1_best=ep[vf.index(max(vf))],
        pertask_min=pertask_min,
        t_lossp2_true=tf[lt], t_lossp2_last=tf[lk], t_f1p3=tf[fb], cens=(ct,ck,cf),
        e_lossp2_true=ep[lt], e_lossp2_last=ep[lk], e_f1p3=ep[fb],
        true=true, kl=kl))
json.dump(rows,open("stopping_rule_check_rows.json","w"))
one=[r for r in rows if r["epe"]==1]; four=[r for r in rows if r["epe"]>1]
print(f"{'run':52} n  trueMin lastMin F1best | test F1: lossp2(true) lossp2(last) F1p3")
for r in rows:
    c="".join("*" if x else "." for x in r["cens"])
    print(f"{r['run'][:52]:52} {r['n']:2} {r['true_min']:6g} {r['kl_min']:7g} {r['f1_best']:6g} | {r['t_lossp2_true']:6.1f}@{r['e_lossp2_true']:<5g} {r['t_lossp2_last']:6.1f}@{r['e_lossp2_last']:<5g} {r['t_f1p3']:6.1f}@{r['e_f1p3']:<5g} {c}")
for lab,S in (("1 scoring/epoch",one),("4 scorings/epoch",four)):
    if not S: continue
    print(f"\n== {lab}: {len(S)} runs")
    print(" true-loss min epoch:", sorted(r['true_min'] for r in S))
    print(" last-batch min epoch:", sorted(r['kl_min'] for r in S))
    print(" val-F1 best epoch:  ", sorted(r['f1_best'] for r in S))
    pt=[e for r in S for e in r['pertask_min']]
    from collections import Counter; print(" per-task loss min epochs:", sorted(Counter(pt).items()))
    for k in ("t_lossp2_true","t_lossp2_last","t_f1p3"): print(f"  {k}: {st.mean(r[k] for r in S):.2f}")
    d1=[r['t_f1p3']-r['t_lossp2_true'] for r in S]; d2=[r['t_f1p3']-r['t_lossp2_last'] for r in S]; d3=[r['t_lossp2_true']-r['t_lossp2_last'] for r in S]
    for lab2,d in (("F1p3 - lossp2(true)",d1),("F1p3 - lossp2(last)",d2),("lossp2 true - last",d3)):
        m=st.mean(d); s=st.stdev(d)/len(d)**.5
        print(f"  {lab2}: {m:+.2f}  [{m-2.0*s:+.2f}, {m+2.0*s:+.2f}]  wins {sum(x>0 for x in d)}/{len(d)} ties {sum(x==0 for x in d)}")
    same=sum(r['e_lossp2_true']==r['e_lossp2_last'] for r in S); print(f"  loss p2 picks same epoch true vs last: {same}/{len(S)}")
