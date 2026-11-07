"""Run both pipelines offline with a scripted LLM and an in-memory BM25 index."""

import json
import re

import yaml

from dmlab_rag.generation.pipeline import generate
from dmlab_rag.io import answer_record, validate_answers, write_answers
from dmlab_rag.retrieval import Doc
from dmlab_rag.retrieval.pipeline import retrieve
from dmlab_rag.text import tokens, word_count
from dmlab_rag.topics import Topic

CORPUS = {
    "d1": "Solar panels cut bakery electricity bills by thirty percent in Ohio. Installers report a "
          "typical payback period of seven years for small shops with ovens running all day.",
    "d2": "Federal tax credits cover thirty percent of solar installation costs for small businesses. "
          "Several states add rebates that shorten the payback period further for owners.",
    "d3": "Battery storage lets bakeries run ovens before sunrise using stored solar energy. "
          "Lithium systems add cost but reduce peak demand charges from the utility company.",
    "d4": "River trout populations declined after the dam was built upstream last decade. "
          "Anglers reported fewer fish during the spring season across the whole valley.",
}


class FakeSearcher:
    def search(self, query, k):
        q = set(tokens(query))
        scored = sorted(CORPUS, key=lambda d: -len(q & set(tokens(CORPUS[d]))))
        return [Doc(d, CORPUS[d]) for d in scored[:k] if q & set(tokens(CORPUS[d]))]


def fake_llm(prompt, system=None):
    if "information retrieval planner" in prompt:
        return json.dumps({"subquestions": ["solar payback bakery"], "entity_terms": ["Ohio"],
                           "keywords": ["solar", "bakery", "tax credits", "battery"],
                           "sparse_big_query": "solar bakery payback tax credits battery"})
    if "evidence sketches" in prompt:
        return json.dumps({"evidence_sketches": ["Solar panels pay back in seven years for bakeries.",
                                                 "Tax credits cover thirty percent of costs."]})
    if "atomic sub-narratives" in prompt and "Decompose" in prompt:
        return json.dumps({"sub_narratives": [{"text": "solar payback period", "importance": "vital"},
                                              {"text": "tax credits", "importance": "vital"},
                                              {"text": "battery storage", "importance": "okay"}]})
    if "strict TREC passage relevance assessor" in prompt:
        n = len(re.findall(r"^\[\d+\]", prompt.split("Passages:")[1], re.M))
        return json.dumps({"grades": [{"id": i, "score": 80 - 10 * i} for i in range(n)]})
    if "##final score" in prompt:
        return "##final score: 0" if "trout" in prompt else "##final score: 3"
    if "listwise reranker" in prompt:
        n = len(re.findall(r"^\[\d+\]", prompt.split("Passages:")[1], re.M))
        return json.dumps({"ordered_ids": list(reversed(range(n)))})
    if prompt.startswith("Extract up to"):
        ev = re.findall(r"^\[(\d+)\] (.+)$", prompt.split("Evidence:")[1], re.M)
        return json.dumps({"claims": [{"text": t[:120], "support": [int(i)]} for i, t in ev[:5]]})
    if prompt.startswith("Convert each claim"):
        claims = json.loads(prompt.split("Claims: ")[1].split("\nReturn JSON")[0])
        return json.dumps({"sentences": [{"id": c["id"], "text": c["claim"],
                                          "citations": c["allowed_citations"]} for c in claims]})
    if "strict TREC RAG support assessor" in prompt:
        n = len(re.findall(r"^\d+\. ", prompt.split("SENTENCES:")[1], re.M))
        return json.dumps({"labels": ["support"] * n})
    raise AssertionError(f"unexpected prompt: {prompt[:80]}")


def test_pipeline_end_to_end(tmp_path):
    cfg = yaml.safe_load(open("configs/submitted.yaml"))
    cfg["retrieval"]["subnarrative_examples"] = None
    models = {role: fake_llm for role in cfg["models"]}
    topic = Topic("T1", "I run a bakery and want to know if solar panels pay off, what tax credits "
                        "exist, and whether battery storage is worth it.")

    ret = retrieve(topic, FakeSearcher(), models, cfg)
    assert "d4" not in [d.docid for d in ret.submitted]        # irrelevant doc cut by depth policy
    assert 1 <= len(ret.submitted) <= 100

    ans = generate(ret, models, cfg)
    assert ans.answer and ans.length <= 1000
    rec = answer_record(ans, "DMLabRAGMerge", "DM-Lab-RAG-2026")
    assert rec["metadata"]["narrative"] == topic.narrative
    assert all(len(a["citations"]) <= 3 for a in rec["answer"])
    assert word_count(" ".join(a["text"] for a in rec["answer"])) == rec["response_length"]

    path = tmp_path / "answers.jsonl"
    write_answers([ans], path, "DMLabRAGMerge", "DM-Lab-RAG-2026")
    assert validate_answers(path, [topic]) == []
