import pytest

from dmlab_rag.eval.metrics import citation_precision_recall, ndcg_at_k, paired_bootstrap
from dmlab_rag.generation.assemble import assemble, sentence_score
from dmlab_rag.generation.citations import repair
from dmlab_rag.generation.claims import Claim, ClaimPool, windows
from dmlab_rag.generation.evidence import harvest
from dmlab_rag.generation.sentences import Sentence, allowed_citations
from dmlab_rag.llm import parse_json
from dmlab_rag.retrieval import Doc


def test_harvest_round_robin():
    d1 = Doc("d1", "Solar panels cut bakery electricity bills by thirty percent in Ohio. "
                   "Installers report a typical payback period of seven years for small shops.")
    d2 = Doc("d2", "Federal tax credits cover thirty percent of solar installation costs today. "
                   "Unrelated sentence about the weather patterns over the ocean this week.")
    docs, ev = harvest([d1, d2], ["solar bakery payback tax credits"], max_sentences=3)
    assert len(ev) == 3
    assert [e.doc_index for e in ev[:2]] == [0, 1]          # alternates across documents


def test_claim_pool_dedup_and_cap():
    pool = ClaimPool(max_claims=2, dup_overlap=0.85)
    pool.add([Claim(-1, "Solar payback period is seven years", [0], "pass1"),
              Claim(-1, "solar payback period is seven years", [1], "window"),
              Claim(-1, "Tax credits cover thirty percent", [2], "window"),
              Claim(-1, "Batteries store power overnight", [3], "window")])
    assert [c.cid for c in pool.claims] == [0, 1]
    assert pool.claims[1].text.startswith("Tax")


def test_windows_cover_all_evidence():
    ev = list(range(40))
    ws = windows(ev, size=18, stride=9)
    assert all(len(w) == 18 for w in ws)
    assert set().union(*map(set, ws)) == set(ev)


def test_allowed_citations_unique_docs():
    from dmlab_rag.generation.evidence import Evidence
    ev = [Evidence(0, 2, "x", "", 1), Evidence(1, 2, "x", "", 1), Evidence(2, 5, "y", "", 1)]
    assert allowed_citations(Claim(0, "c", [1, 0, 2], "pass1"), ev) == [2, 5]


def test_sentence_score_eq5():
    # novelty 2 ({b, c}); aspect {a, b}: coverage goes 1/2 -> 1, gain 0.5, weighted 1.5
    assert sentence_score({"a", "b", "c"}, {"a"}, [{"a", "b"}]) == pytest.approx(3.5)


def test_assemble_respects_budget_and_stops_at_zero():
    s = [Sentence(0, "solar payback seven years", [0]),
         Sentence(1, "solar payback seven years", [1]),        # adds nothing once the first is in
         Sentence(2, "tax credits thirty percent", [0])]
    out = assemble(s, ["tax credits", "payback period"], budget=8)
    assert {x.cid for x in out} == {0, 2} or {x.cid for x in out} == {1, 2}
    assert sum(len(x.text.split()) for x in out) <= 8


def test_repair_drops_not_support_only():
    s = [Sentence(0, "a", [0, 1]), Sentence(1, "b", [1])]
    labels = {(0, 0): "support", (0, 1): "not_support", (1, 1): "partial_support"}
    out = repair(s, labels)
    assert out[0].citations == [0] and out[1].citations == [1]


def test_citation_metrics_eq1_eq2():
    cited = [["d1", "d2"], [], ["d3"]]
    support = {(0, "d1"): 1.0, (0, "d2"): 0.5, (2, "d3"): 0.0}
    p, r = citation_precision_recall(cited, support)
    assert p == pytest.approx(1.5 / 3)
    assert r == pytest.approx((0.75 + 0 + 0) / 3)


def test_ndcg_and_bootstrap():
    assert ndcg_at_k(["a", "b"], {"a": 3, "b": 1}, 2) == pytest.approx(1.0)
    assert ndcg_at_k(["b", "a"], {"a": 3, "b": 1}, 2) < 1.0
    bs = paired_bootstrap([0.5] * 10, [0.6] * 10, n=500)
    assert bs["mean"] == pytest.approx(0.1) and bs["improved"] == 10


def test_parse_json_tolerates_fences():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! {"ordered_ids": [2, 0]} hope that helps') == {"ordered_ids": [2, 0]}
