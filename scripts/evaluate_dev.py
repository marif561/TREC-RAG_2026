"""Development evaluation: mean nDCG@k across several qrels, plus a paired bootstrap.

    python scripts/evaluate_dev.py --run runs/ensemble.trec --compare runs/listwise.trec \
        --qrels data/dev/qrels.gpt55.txt data/dev/qrels.ministral.txt data/dev/qrels.qwen.txt
"""

import argparse
from collections import defaultdict
from pathlib import Path

from dmlab_rag.eval.metrics import load_qrels, ndcg_at_k, paired_bootstrap


def load_run(path):
    run = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        t, _, d, rank, *_ = line.split()
        run[t].append((int(rank), d))
    return {t: [d for _, d in sorted(v)] for t, v in run.items()}


def per_topic(run, qrels_list, k):
    topics = sorted(set(run) & set.intersection(*(set(q) for q in qrels_list)))
    return topics, [sum(ndcg_at_k(run[t], q[t], k) for q in qrels_list) / len(qrels_list) for t in topics]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--compare", help="second run for a paired bootstrap (compare - run)")
    ap.add_argument("--qrels", nargs="+", required=True)
    ap.add_argument("--k", type=int, nargs="+", default=[10, 30])
    args = ap.parse_args()

    qrels_list = [load_qrels(p) for p in args.qrels]
    base = load_run(args.run)
    other = load_run(args.compare) if args.compare else None
    for k in args.k:
        topics, a = per_topic(base, qrels_list, k)
        line = f"nDCG@{k}: {sum(a) / len(a):.3f} over {len(topics)} topics"
        if other:
            _, b = per_topic({t: other[t] for t in topics}, qrels_list, k)
            bs = paired_bootstrap(a, b)
            line += (f" | compare {sum(b) / len(b):.3f}, diff {bs['mean']:+.3f} "
                     f"[{bs['lo']:+.3f}, {bs['hi']:+.3f}], improved {bs['improved']}/{bs['topics']}")
        print(line)


if __name__ == "__main__":
    main()
