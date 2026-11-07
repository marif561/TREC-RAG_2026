import pytest

from dmlab_rag.retrieval import Doc
from dmlab_rag.retrieval.dedup import DuplicateFilter
from dmlab_rag.retrieval.depth import select_depth
from dmlab_rag.retrieval.ensemble import ensemble_scores, redistributed_weights
from dmlab_rag.retrieval.fusion import rrf
from dmlab_rag.retrieval.judges import parse_umbrela
from dmlab_rag.retrieval.rerank import interpolate
from dmlab_rag.retrieval.subnarratives import SubNarrative, merge_decompositions

LONG = " ".join(f"solar panels reduce bills for bakery number {i} in region r{i % 7}" for i in range(25))


def test_dedup_prefix_and_jaccard():
    f = DuplicateFilter()
    a = Doc("a", LONG)
    b = Doc("b", LONG + " extra tail")                   # same first 200 chars
    c = Doc("c", "Intro. " + LONG)                       # different prefix, near-identical tokens
    d = Doc("d", "completely different text about rivers and fish " * 20)
    out = f.filter([a, b, c, d])
    assert [x.docid for x in out] == ["a", "d"]
    assert f.canonical["c"] == "a"


def test_rrf_matches_formula():
    lists = {"q1": [Doc("x", ""), Doc("y", "")], "q2": [Doc("y", ""), Doc("z", "")]}
    pool = rrf(lists, k=60, pool_size=10)
    assert [d.docid for d in pool] == ["y", "x", "z"]
    assert pool[0].signals["rrf"] == pytest.approx(1 / 62 + 1 / 61)


def test_weight_redistribution_without_coverage():
    w = redistributed_weights({"r", "g", "uq", "um"})
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["uq"] == pytest.approx(0.4 / 0.9)
    assert "cov" not in w


def test_ensemble_prefers_judged_relevant():
    pool = [Doc("top_fused", "", {"g": 10, "uq": 0, "um": 0}),
            Doc("relevant", "", {"g": 90, "uq": 4, "um": 4})]
    ranked = ensemble_scores(pool)
    assert ranked[0].docid == "relevant"
    assert 0 <= ranked[-1].signals["win"] <= 1


def test_listwise_interpolation():
    head = [Doc(str(i), "") for i in range(3)]
    out = interpolate(head, order=[2, 1, 0], alpha=0.55)       # listwise fully reverses ensemble
    assert [d.docid for d in out] == ["2", "1", "0"]            # 0.55 > 0.45, so listwise wins
    assert out[0].signals["final"] == pytest.approx(0.55)


def test_depth_floor_and_threshold():
    ranked = [Doc("a", "", {"uq": 1, "um": 3}), Doc("b", "", {"uq": 1, "um": 1}), Doc("c", "", {"uq": 2})]
    assert [d.docid for d in select_depth(ranked)] == ["a", "c"]
    none = [Doc("a", "", {"uq": 0, "um": 1}), Doc("b", "", {"uq": 1, "um": 0})]
    assert [d.docid for d in select_depth(none, floor=1)] == ["a"]


def test_parse_umbrela():
    assert parse_umbrela("reasoning...\n##final score: 3") == 3
    assert parse_umbrela("no score here") is None


def test_merge_decompositions():
    a = [SubNarrative("solar tax credits", "okay"), SubNarrative("payback period", "vital")]
    b = [SubNarrative("Solar tax credits", "vital"), SubNarrative("battery storage", "okay")]
    merged = merge_decompositions([a, b])
    assert [s.text for s in merged] == ["solar tax credits", "payback period", "battery storage"]
    assert merged[0].importance == "vital"
