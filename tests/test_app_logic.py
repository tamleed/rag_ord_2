import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unittest.mock import patch

import app
from rag_core import Answer


def make_candidate(
    answer_id: int,
    text: str,
    *,
    score_final: float,
    score_dense_raw: float,
    score_dense: float | None = None,
    score_sparse: float = 0.0,
    score_sparse_raw: float = 0.0,
):
    answer = Answer(
        id=answer_id,
        text=text,
        question_variants=[f"variant-{answer_id}"],
        categories=["test"],
        is_active=True,
    )
    return app.RetrievedAnswer(
        answer=answer,
        score_dense=score_dense if score_dense is not None else score_final,
        score_sparse=score_sparse,
        score_final=score_final,
        payload={},
        score_dense_raw=score_dense_raw,
        score_sparse_raw=score_sparse_raw,
    )


def test_exact_raw_match_returns_faq_answer_without_rag():
    question = "exact raw question"
    exact = Answer(id=101, text="faq answer", question_variants=[question], categories=["faq"], is_active=True)

    with patch.object(app, "_match_exact_raw_question", return_value=exact), patch.object(app, "retrieve_answers") as retrieve_mock:
        result = app.answer_question_logic_v2(question)

    retrieve_mock.assert_not_called()
    assert result["answer_source"] == "faq_exact"
    assert result["answer_id"] == 101
    assert result["answer"] == "faq answer"


def test_high_relevance_non_exact_uses_rag_not_no_answer():
    candidate = make_candidate(1, "candidate answer", score_final=0.83, score_dense_raw=0.83)

    with (
        patch.object(app, "_match_exact_raw_question", return_value=None),
        patch.object(app, "retrieve_answers", return_value=[candidate]),
        patch.object(app, "_candidate_has_query_coverage", return_value=False),
        patch.object(app, "_best_variant_overlap", return_value=0.05),
        patch.object(app, "call_llm_v2", return_value="llm answer"),
        patch.object(app, "build_context_fragments", return_value="ctx"),
    ):
        answer_text, answer_ids, _ = app.answer_question_with_rag_v2("non exact question")

    assert answer_text == "llm answer"
    assert answer_ids == [1]


def test_debug_candidates_include_normalized_and_raw_scores():
    candidate = make_candidate(
        1,
        "candidate answer",
        score_final=0.83,
        score_dense_raw=0.86,
        score_dense=1.0,
        score_sparse=0.6,
        score_sparse_raw=0.55,
    )

    with (
        patch.object(app, "_match_exact_raw_question", return_value=None),
        patch.object(app, "answer_question_with_rag_v2", return_value=("llm answer", [1], [candidate])),
    ):
        result = app.answer_question_logic_v2("non exact question")

    debug_candidate = result["debug_candidates"][0]
    assert debug_candidate["score_dense"] == 1.0
    assert debug_candidate["score_dense_normalized"] == 1.0
    assert debug_candidate["score_dense_raw"] == 0.86
    assert debug_candidate["score_sparse_normalized"] == 0.6
    assert debug_candidate["score_sparse_raw"] == 0.55
    assert debug_candidate["score_final_normalized"] == 0.83


def test_score_final_one_reads_directly_from_base_candidate():
    first = make_candidate(1, "first answer", score_final=1.0, score_dense_raw=0.83)
    second = make_candidate(2, "second answer", score_final=1.0, score_dense_raw=0.83)

    with (
        patch.object(app, "_match_exact_raw_question", return_value=None),
        patch.object(app, "retrieve_answers", return_value=[first, second]),
        patch.object(app, "call_llm_v2") as llm_mock,
    ):
        answer_text, answer_ids, _ = app.answer_question_with_rag_v2("non exact question")

    llm_mock.assert_not_called()
    assert answer_text == "first answer"
    assert answer_ids == [1]


def test_low_relevance_returns_no_answer_only_after_model_confirmation():
    candidate = make_candidate(1, "candidate answer", score_final=0.4, score_dense_raw=0.1)

    with (
        patch.object(app, "_match_exact_raw_question", return_value=None),
        patch.object(app, "retrieve_answers", return_value=[candidate]),
        patch.object(app, "_candidate_has_query_coverage", return_value=False),
        patch.object(app, "_best_variant_overlap", return_value=0.05),
        patch.object(app, "call_llm_v2", return_value=app.NO_ANSWER_TEXT),
    ):
        answer_text, answer_ids, _ = app.answer_question_with_rag_v2("irrelevant question")

    assert answer_text == app.NO_ANSWER_TEXT
    assert answer_ids is None


def test_low_relevance_with_unavailable_model_returns_service_message():
    candidate = make_candidate(1, "candidate answer", score_final=0.4, score_dense_raw=0.1)

    with (
        patch.object(app, "_match_exact_raw_question", return_value=None),
        patch.object(app, "retrieve_answers", return_value=[candidate]),
        patch.object(app, "_candidate_has_query_coverage", return_value=False),
        patch.object(app, "_best_variant_overlap", return_value=0.05),
        patch.object(app, "call_llm_v2", return_value="Ошибка: timeout"),
    ):
        result = app.answer_question_logic_v2("irrelevant question")

    assert result["answer"] == app.MODEL_UNAVAILABLE_TEXT
    assert result["answer_source"] == "rag_llm"
    assert result["answer_id"] is None


def test_close_top_candidates_return_multiple_answers_message():
    first = make_candidate(1, "first answer", score_final=0.8, score_dense_raw=0.8)
    second = make_candidate(2, "second answer", score_final=0.77, score_dense_raw=0.77)
    third = make_candidate(3, "third answer", score_final=0.75, score_dense_raw=0.75)
    fourth = make_candidate(4, "fourth answer", score_final=0.35, score_dense_raw=0.35)

    with (
        patch.object(app, "_match_exact_raw_question", return_value=None),
        patch.object(app, "retrieve_answers", return_value=[first, second, third, fourth]),
        patch.object(app, "_candidate_has_query_coverage", return_value=True),
        patch.object(app, "_best_variant_overlap", return_value=0.6),
    ):
        answer_text, answer_ids, _ = app.answer_question_with_rag_v2("ambiguous question")

    assert "Пожалуйста, используйте тот, который точнее соответствует вашему контексту." in answer_text
    assert "Ответ 3:\nthird answer" in answer_text
    assert answer_ids == [1, 2, 3]


def test_more_than_four_close_candidates_asks_to_clarify():
    candidates = [
        make_candidate(1, "first answer", score_final=0.8, score_dense_raw=0.8),
        make_candidate(2, "second answer", score_final=0.79, score_dense_raw=0.79),
        make_candidate(3, "third answer", score_final=0.78, score_dense_raw=0.78),
        make_candidate(4, "fourth answer", score_final=0.77, score_dense_raw=0.77),
        make_candidate(5, "fifth answer", score_final=0.76, score_dense_raw=0.76),
    ]

    with (
        patch.object(app, "_match_exact_raw_question", return_value=None),
        patch.object(app, "retrieve_answers", return_value=candidates),
        patch.object(app, "_candidate_has_query_coverage", return_value=True),
        patch.object(app, "_best_variant_overlap", return_value=0.6),
    ):
        result = app.answer_question_logic_v2("ambiguous question")

    assert result["answer"] == app.TOO_MANY_RELEVANT_ANSWERS_TEXT
    assert result["answer_source"] == "rag_llm"
