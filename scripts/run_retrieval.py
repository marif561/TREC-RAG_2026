"""Produce the DMLabWin055 retrieval run (and cache full rankings for generation).

    python scripts/run_retrieval.py --config configs/submitted.yaml --out runs/
"""

import argparse
import json
from pathlib import Path

import yaml
from tqdm import tqdm

from dmlab_rag.io import write_trec_run
from dmlab_rag.llm import build_models
from dmlab_rag.retrieval.pipeline import retrieve
from dmlab_rag.retrieval.search import make_searcher
from dmlab_rag.topics import load_topics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/submitted.yaml")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--limit", type=int, help="only the first N topics (smoke test)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    topics = load_topics(cfg["topics"]["path"], cfg["topics"].get("sha256_prefix"))[: args.limit]
    searcher, models = make_searcher(cfg["search"]), build_models(cfg)
    out = Path(args.out)
    (out / "retrieval_cache").mkdir(parents=True, exist_ok=True)

    submitted = {}
    for t in tqdm(topics, desc="DMLabWin055"):
        res = retrieve(t, searcher, models, cfg)
        submitted[t.topic_id] = res.submitted
        (out / "retrieval_cache" / f"{t.topic_id}.json").write_text(json.dumps({
            "topic_id": t.topic_id, "narrative": t.narrative, "views": res.views,
            "sub_narratives": [s.__dict__ for s in res.sub_narratives],
            "reranked": [{"docid": d.docid, "text": d.text, "signals": d.signals} for d in res.reranked],
            "submitted": [d.docid for d in res.submitted],
        }, ensure_ascii=False))

    write_trec_run(submitted, out / f"{cfg['retrieval_run_id']}.trec", cfg["retrieval_run_id"])
    depths = sorted(len(v) for v in submitted.values())
    print(f"{len(depths)} topics | depth min {depths[0]} median {depths[len(depths) // 2]} "
          f"mean {sum(depths) / len(depths):.1f} max {depths[-1]}")


if __name__ == "__main__":
    main()
