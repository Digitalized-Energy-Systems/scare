"""Diff two repro_nondet snapshot files; report the first divergent step
and classify what diverged (message ordering vs physical/solver values)."""

from __future__ import annotations

import json
import sys


def load(p):
    with open(p) as fh:
        return json.load(fh)


def queue_key_order(q):
    # ordering signature: (sender, receiver, ctype) in stored order
    return [(m["snd"], m["rcv"], m["ctype"]) for m in q]


def queue_multiset(q):
    from collections import Counter

    return Counter((m["snd"], m["rcv"], m["ctype"], m["dt"]) for m in q)


def monee_diff(a, b):
    diffs = []
    for kind in ("childs", "branches", "nodes"):
        am, bm = a.get(kind, {}), b.get(kind, {})
        for k in sorted(set(am) | set(bm)):
            av, bv = am.get(k, {}), bm.get(k, {})
            for field in sorted(set(av) | set(bv)):
                x, y = av.get(field), bv.get(field)
                if x != y:
                    diffs.append(f"{kind}[{k}].{field}: {x!r} != {y!r}")
    return diffs


def main():
    a = load(sys.argv[1])
    b = load(sys.argv[2])
    n = min(len(a), len(b))
    print(f"A={len(a)} snaps  B={len(b)} snaps  comparing {n}")
    for i in range(n):
        sa, sb = a[i], b[i]
        label = sa.get("label", sa.get("failures", f"idx{i}"))
        if sa == sb:
            continue
        print("\n=== FIRST DIVERGENCE ===")
        print(f"snapshot index {i}  label={label!r}")
        for key in ("t", "step", "delivered", "msg_seq", "n_recorded"):
            if sa.get(key) != sb.get(key):
                print(f"  {key}: A={sa.get(key)!r}  B={sb.get(key)!r}")

        # classify
        qa, qb = sa.get("queue", []), sb.get("queue", [])
        if qa != qb:
            msa, msb = queue_multiset(qa), queue_multiset(qb)
            if msa == msb:
                print("  QUEUE: same multiset, DIFFERENT ORDER -> ordering nondet")
            else:
                print("  QUEUE: different multiset -> different messages sent")
            ka, kb = queue_key_order(qa), queue_key_order(qb)
            print(f"  queue len A={len(qa)} B={len(qb)}")
            for j in range(min(len(ka), len(kb))):
                if ka[j] != kb[j]:
                    print(f"  first queue slot differing: idx {j}")
                    print(f"    A: {qa[j]}")
                    print(f"    B: {qb[j]}")
                    break
            # show full short order if small
            if len(ka) <= 30:
                print(f"  A order: {ka}")
                print(f"  B order: {kb}")

        if sa.get("inbox") != sb.get("inbox"):
            print(f"  INBOX: A={sa.get('inbox')}  B={sb.get('inbox')}")

        md = monee_diff(sa.get("monee", {}), sb.get("monee", {}))
        if md:
            print(f"  MONEE physical-state diffs ({len(md)}):")
            for line in md[:20]:
                print(f"    {line}")
        return
    print("\nNo divergence in compared snapshots (deterministic over this range).")


if __name__ == "__main__":
    main()
