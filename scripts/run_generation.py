"""Produce the DMLabRAGMerge answer file from cached retrieval results.

    python scripts/run_generation.py --config configs/submitted.yaml --runs runs/
"""

import argparse
import json
from pathlib import Path

import yaml
from tqdm import tqdm

from dmlab_rag.generation.pipeline import generate
from dmlab_rag.io import validate_answers, write_answers
from dmlab_rag.llm import build_models
from dmlab_rag.retrieval import Doc
from dmlab_rag.retrieval.pipeline import RetrievalResult
from dmlab_rag.retrieval.subnarratives import SubNarrative
from dmlab_rag.topics import Topic, load_topics


def load_cached(path: Path) -> RetrievalResult:
    c = json.loads(path.read_text(encoding="utf-8"))
    docs = [Doc(d["docid"], d["text"], d["signals"]) for d in c["reranked"]]
    keep = set(c["submitted"])
    return RetrievalResult(Topic(c["topic_id"], c["narrative"]),
                           [SubNarrative(**s) for s in c["sub_narratives"]],
                           docs, [d for d in docs if d.docid in keep], c["views"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/submitted.yaml")
    ap.add_argument("--runs", default="runs")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    models = build_models(cfg)
    runs = Path(args.runs)
    answers = [generate(load_cached(p), models, cfg)
               for p in tqdm(sorted((runs / "retrieval_cache").glob("*.json")), desc="DMLabRAGMerge")]

    out = runs / f"{cfg['generation_run_id']}.jsonl"
    write_answers(answers, out, cfg["generation_run_id"], cfg["team_id"])
    topics = load_topics(cfg["topics"]["path"])
    topics = [t for t in topics if t.topic_id in {a.topic_id for a in answers}]
    problems = validate_answers(out, topics)
    print("\n".join(problems) if problems else f"OK: {len(answers)} answers -> {out}")


if __name__ == "__main__":
    main()
