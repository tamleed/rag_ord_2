import os
from typing import List, Dict, Optional, Set

import requests
from fastapi import FastAPI, Depends, HTTPException, status, Header
from pydantic import BaseModel

from qdrant_client import QdrantClient, models

from rag_core import (
    Answer,
    load_answers,
    build_normalized_q_index,
    build_raw_q_index,
    load_tfidf_vocab,
    normalize,
    expand_with_synonyms,
    compute_sparse_vector,
    dense_embed_query,
)

# ---------- Конфиг ----------

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QA_DB_PATH = os.getenv("QA_DB_PATH", "./qa_db.json")
TFIDF_VOCAB_PATH = os.getenv("TFIDF_VOCAB_PATH", "./tfidf_vocab.json")
QA_DB_PATH_V2 = os.getenv("QA_DB_PATH_V2", "./qa_db_merged.json")
TFIDF_VOCAB_PATH_V2 = os.getenv("TFIDF_VOCAB_PATH_V2", TFIDF_VOCAB_PATH)
COLLECTION_NAME = "faq"
QDRANT_EXACT_SEARCH = os.getenv("QDRANT_EXACT_SEARCH", "1") == "1"

ALPHA_DENSE = 0.6          # вес dense-вектора
SCORE_THRESHOLD = 0.2      # порог уверенности retriever'а для относительного hybrid-score
LOW_RELEVANCE_DENSE_THRESHOLD = 0.2  # абсолютный порог cosine similarity для dense-совпадения
LOW_RELEVANCE_OVERLAP_THRESHOLD = 0.2  # минимальное лексическое пересечение для осмысленного матча
SCORE_THRESHOLD_V1 = float(os.getenv("SCORE_THRESHOLD_V1", "0.2"))
TOP1_GAP_THRESHOLD = 0.08  # минимальный отрыв top-1 от top-2
MULTI_ANSWER_GAP_THRESHOLD = 0.03  # насколько близки score top-1 и top-2
MULTI_ANSWER_THIRD_GAP_THRESHOLD = 0.15  # насколько top-3 отстаёт от top-1
MULTI_ANSWER_RELATIVE_RATIO = 0.7  # близкие кандидаты: не хуже 70% от score top-1
MULTI_ANSWER_NEXT_FAR_RATIO = 0.5  # следующий кандидат считается далеким, если ниже 50% от score top-1
MAX_MULTI_ANSWER_CANDIDATES = 4

# локальная LLM
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "")
LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "")
  # API-ключ для доступа к самому RAG-API
RAG_API_KEY = os.getenv("RAG_API_KEY", "")
LLM_MAX_TOKENS_V2 = int(os.getenv("LLM_MAX_TOKENS_V2", "1200"))


# ---------- Глобальное состояние ----------

client_qdrant: QdrantClient = QdrantClient(url=QDRANT_URL)

ANSWERS: List[Answer] = load_answers(QA_DB_PATH)
ANSWERS_BY_ID: Dict[int, Answer] = {a.id: a for a in ANSWERS}

QUESTION_INDEX_RAW: Dict[str, List[int]] = build_raw_q_index(ANSWERS)
QUESTION_INDEX_NORMALIZED: Dict[str, List[int]] = build_normalized_q_index(ANSWERS)

TERM2ID, IDF = load_tfidf_vocab(TFIDF_VOCAB_PATH)

# Отдельные ресурсы для v2 (по умолчанию совпадают с v1, если env не задан).
ANSWERS_V2: List[Answer] = load_answers(QA_DB_PATH_V2)
ANSWERS_BY_ID_V2: Dict[int, Answer] = {a.id: a for a in ANSWERS_V2}

QUESTION_INDEX_RAW_V2: Dict[str, List[int]] = build_raw_q_index(ANSWERS_V2)
QUESTION_INDEX_NORMALIZED_V2: Dict[str, List[int]] = build_normalized_q_index(ANSWERS_V2)

TERM2ID_V2, IDF_V2 = load_tfidf_vocab(TFIDF_VOCAB_PATH_V2)


# ---------- Авторизация по API-ключу ----------

async def verify_api_key(x_api_key: str = Header(None)):
    """
    Простейшая защита: все запросы к /answer и /answer_full
    должны содержать заголовок X-API-Key с правильным RAG_API_KEY.
    Если RAG_API_KEY не задан (dev), проверка отключается.
    """
    if not RAG_API_KEY:
        # dev-режим: ключ не настроен — пропускаем без проверки
        return
    if x_api_key != RAG_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )





# ---------- Pydantic-модели API ----------

class QueryRequest(BaseModel):
    question: str
    debug: bool = False


class QueryResponse(BaseModel):
    answer: str
    answer_source: str  # "faq_exact" | "rag_llm" | "no_answer"
    answer_id: Optional[int]
    debug: Optional[Dict] = None


# ---------- Внутренний класс для кандидатов ----------

class RetrievedAnswer:
    def __init__(
        self,
        answer: Answer,
        score_dense: float,
        score_sparse: float,
        score_final: float,
        payload: Dict,
        score_dense_raw: float = 0.0,
        score_sparse_raw: float = 0.0,
    ):
        self.answer = answer
        self.score_dense = score_dense
        self.score_sparse = score_sparse
        self.score_final = score_final
        self.payload = payload
        self.score_dense_raw = score_dense_raw
        self.score_sparse_raw = score_sparse_raw


NO_ANSWER_TEXT = (
    "В нашей базе знаний нет точного ответа на этот вопрос. "
    "Пожалуйста, уточните формулировку или обратитесь к специалисту ОРД."
)
MODEL_UNAVAILABLE_TEXT = (
    "Не удалось подтвердить отсутствие точного ответа, потому что модель временно недоступна. "
    "Пожалуйста, повторите запрос."
)
TOO_MANY_RELEVANT_ANSWERS_TEXT = (
    "Количество релевантных ответов больше четырех, уточните, пожалуйста, вопрос."
)

RUS_STOPWORDS: Set[str] = {
    "и", "в", "во", "на", "по", "к", "ко", "у", "о", "об", "от", "до", "за", "из",
    "под", "над", "для", "а", "но", "или", "ли", "не", "это", "как", "что", "где", "когда",
    "нужно", "надо", "можно", "нельзя", "если", "при", "про", "так", "же", "бы", "быть",
    "я", "мы", "вы", "он", "она", "они", "мой", "ваш", "наш", "меня", "мне", "вам",
}


TOKEN_CANONICAL_MAP: Dict[str, str] = {
    "корректиров": "правк",
    "изменен": "правк",
    "внести": "правк",
    "правки": "правк",
    "креатив": "креатив",
    "ссыл": "ссылк",
    "целевая": "целев",
    "целевых": "целев",
    "адрес": "адрес",
    "площадки": "площадк",
    "площадка": "площадк",
    "сверки": "сверк",
    "сверка": "сверк",
}


def _canonical_token(token: str) -> str:
    for key, value in TOKEN_CANONICAL_MAP.items():
        if token.startswith(key):
            return value
    if len(token) >= 6:
        return token[:5]
    return token


def _informative_tokens(text: str) -> Set[str]:
    tokens = set(normalize(text).split())
    prepared = {t for t in tokens if len(t) >= 3 and t not in RUS_STOPWORDS}
    return {_canonical_token(t) for t in prepared}


def _candidate_has_query_coverage(question: str, candidate: RetrievedAnswer) -> bool:
    """
    Проверяем, что ключевые токены вопроса представлены в top-кандидате.
    Это защищает от ответов на похожий, но другой запрос (например, "акт" vs "акт сверки").
    """
    q_tokens = _informative_tokens(question)
    if not q_tokens:
        return True

    candidate_corpus = " ".join(
        [candidate.answer.text] + candidate.answer.question_variants + candidate.answer.categories
    )
    c_tokens = _informative_tokens(candidate_corpus)

    missing = q_tokens - c_tokens

    # Допускаем потерю 1 токена, если запрос длинный, иначе требуем полное покрытие.
    if len(q_tokens) >= 6:
        return len(missing) <= 1
    return len(missing) == 0




def _token_overlap_score(question: str, variant: str) -> float:
    q_tokens = _informative_tokens(question)
    v_tokens = _informative_tokens(variant)
    if not q_tokens or not v_tokens:
        return 0.0

    inter = len(q_tokens & v_tokens)
    if inter == 0:
        return 0.0

    jaccard = inter / len(q_tokens | v_tokens)
    query_coverage = inter / len(q_tokens)
    return max(jaccard, query_coverage)


def find_best_faq_match(question: str, threshold: float = 0.3) -> Optional[int]:
    """
    Лексический матч по вариантам вопросов, чтобы находить
    переформулированные, но точные пользовательские запросы.
    """
    best_id: Optional[int] = None
    best_score = 0.0

    for ans in ANSWERS_V2:
        for variant in ans.question_variants:
            score = _token_overlap_score(question, variant)
            if score > best_score:
                best_score = score
                best_id = ans.id

    if best_score >= threshold:
        q_tokens = _informative_tokens(question)
        # Для общего вопроса про правки креатива приоритетно отдаём расширенный ответ (id=32).
        if best_id == 15 and {"правк", "креатив"}.issubset(q_tokens) and 32 in ANSWERS_BY_ID_V2:
            return 32
        # Для вопроса о нескольких целевых ссылках приоритетно отдаём профильный ответ (id=49).
        if {"целев", "ссылк"}.issubset(q_tokens) and 49 in ANSWERS_BY_ID_V2:
            return 49
        return best_id
    return None


def _is_confident_top_candidate(candidates: List[RetrievedAnswer]) -> bool:
    if not candidates:
        return False
    if candidates[0].score_final < SCORE_THRESHOLD:
        return False
    if len(candidates) == 1:
        return True
    return (candidates[0].score_final - candidates[1].score_final) >= TOP1_GAP_THRESHOLD


def _best_variant_overlap(question: str, candidate: RetrievedAnswer) -> float:
    """
    Максимальное лексическое пересечение вопроса пользователя
    с вариантами вопросов в карточке ответа.
    """
    if not candidate.answer.question_variants:
        return 0.0
    return max((_token_overlap_score(question, v) for v in candidate.answer.question_variants), default=0.0)


# ---------- Retriever (hybrid dense + sparse) ----------



def _build_search_params() -> Optional[models.SearchParams]:
    """
    Для одинаковых запросов важно получать одинаковый shortlist кандидатов.
    exact=True отключает ANN-аппроксимацию на стороне Qdrant и убирает
    недетерминизм ближайших соседей при очень близких score.
    """
    if not QDRANT_EXACT_SEARCH:
        return None
    return models.SearchParams(exact=True)
def retrieve_answers(
    question: str,
    top_k: int = 5,
    dense_limit: int = 20,
    sparse_limit: int = 20,
) -> List[RetrievedAnswer]:
    norm_q = normalize(question)

    # 1. dense-вектор
    dense_q = dense_embed_query(question)

    # 2. sparse-вектор
    expanded_q = expand_with_synonyms(norm_q)
    q_indices, q_values = compute_sparse_vector(expanded_q, TERM2ID_V2, IDF_V2)

    query_filter = models.Filter(
        must=[models.FieldCondition(key="is_active", match=models.MatchValue(value=True))]
    )

    # 3. dense-запрос
    search_params = _build_search_params()

    dense_points = client_qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=dense_q,
        using="dense",
        query_filter=query_filter,
        limit=dense_limit,
        with_payload=True,
        search_params=search_params,
    ).points

    # 4. sparse-запрос
    sparse_points = []
    if q_indices:
        sparse_vec = models.SparseVector(indices=q_indices, values=q_values)
        sparse_points = client_qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=sparse_vec,
            using="sparse",
            query_filter=query_filter,
            limit=sparse_limit,
            with_payload=True,
            search_params=search_params,
        ).points

    dense_scores: Dict[int, float] = {}
    sparse_scores: Dict[int, float] = {}

    for p in dense_points:
        dense_scores[int(p.id)] = float(p.score)

    for p in sparse_points:
        sparse_scores[int(p.id)] = float(p.score)

    def normalize_scores(scores: Dict[int, float]) -> Dict[int, float]:
        if not scores:
            return {}
        mx = max(scores.values())
        if mx <= 0:
            return {k: 0.0 for k, v in scores.items()}
        return {k: v / mx for k, v in scores.items()}

    # Важно: после такой нормализации лучший dense/sparse кандидат внутри текущей выборки
    # получает score 1.0, поэтому 1.0 здесь означает "лучший в shortlist", а не "точное совпадение".
    dense_norm = normalize_scores(dense_scores)
    sparse_norm = normalize_scores(sparse_scores)

    all_ids = set(dense_norm.keys()) | set(sparse_norm.keys())
    retrieved: List[RetrievedAnswer] = []
    src_points = {int(p.id): p for p in dense_points}
    src_points.update({int(p.id): p for p in sparse_points})

    for ans_id in all_ids:
        sd = dense_norm.get(ans_id, 0.0)
        ss = sparse_norm.get(ans_id, 0.0)
        sd_raw = dense_scores.get(ans_id, 0.0)
        ss_raw = sparse_scores.get(ans_id, 0.0)
        final = ALPHA_DENSE * sd + (1.0 - ALPHA_DENSE) * ss

        payload = src_points[ans_id].payload or {}
        ans = ANSWERS_BY_ID_V2.get(ans_id)
        if not ans:
            ans = Answer(
                id=ans_id,
                text=payload.get("answer_text", ""),
                question_variants=payload.get("question_variants", []),
                categories=payload.get("categories", []),
                is_active=payload.get("is_active", True),
            )

        retrieved.append(
            RetrievedAnswer(
                answer=ans,
                score_dense=sd,
                score_sparse=ss,
                score_final=final,
                payload=payload,
                score_dense_raw=sd_raw,
                score_sparse_raw=ss_raw,
            )
        )

    # Детерминированная сортировка: одинаковые score не должны давать разный порядок между запросами.
    retrieved.sort(key=lambda r: (-r.score_final, r.answer.id))
    return retrieved[:top_k]


# ---------- LLM / RAG ----------

def build_context_fragments(candidates: List[RetrievedAnswer]) -> str:
    parts: List[str] = []
    for i, cand in enumerate(candidates, start=1):
        ans = cand.answer
        parts.append(f"Фрагмент {i}:")
        if ans.categories:
            parts.append(f"Категории: {', '.join(ans.categories)}")
        if ans.question_variants:
            qs = ", ".join(ans.question_variants[:5])
            parts.append(f"Варианты вопросов: {qs}")
        parts.append("Ответ:")
        parts.append(ans.text)
        parts.append("")
    return "\n".join(parts)


def call_llm_v2(system_prompt: str, user_prompt: str) -> str:
    """
    Вызываем локальную модель по HTTP.

    Формат:
    POST LOCAL_LLM_URL
      headers: x-api-key: LOCAL_LLM_API_KEY
      body: {
        "disable_reasoning": true,
        "messages": [{"role": "user", "content": "<full_prompt>"}],
        "max_tokens": LLM_MAX_TOKENS_V2
      }

    Ожидаем ответ: {"content": "..."}
    """
    if not LOCAL_LLM_URL:
        return "Ошибка: не настроен LOCAL_LLM_URL для локальной модели."

    full_prompt = system_prompt + "\n\n" + user_prompt

    headers = {
        "Content-Type": "application/json",
        "x-api-key": LOCAL_LLM_API_KEY,
    }
    payload = {
        "disable_reasoning": True,
        "messages": [
            {"role": "user", "content": full_prompt}
        ],
        "max_tokens": LLM_MAX_TOKENS_V2,
        "temperature": 0,
        "top_p": 1,
    }

    try:
        resp = requests.post(
            LOCAL_LLM_URL,
            json=payload,
            headers=headers,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data.get("content")
        if not isinstance(content, str):
            return "Ошибка: локальная модель вернула неожиданный формат ответа."
        return content.strip()
    except Exception as e:
        return f"Ошибка при обращении к локальной модели: {e}"




def _match_exact_raw_question(question: str) -> Optional[Answer]:
    """
    100% совпадение вопроса пользователя с вопросом в базе (raw).
    Только в этом случае отвечаем "как есть" из БЗ и не идём в модель.
    """
    raw_q = question.strip()
    direct_ids_raw = QUESTION_INDEX_RAW_V2.get(raw_q)
    if not direct_ids_raw:
        return None
    return ANSWERS_BY_ID_V2[direct_ids_raw[0]]


def _match_normalized_question(question: str) -> Optional[Answer]:
    """
    Неточное (нормализованное) совпадение: отдельная ветка после raw exact.
    """
    norm_q = normalize(question)
    direct_ids_norm = QUESTION_INDEX_NORMALIZED_V2.get(norm_q)
    if not direct_ids_norm:
        return None
    return ANSWERS_BY_ID_V2[direct_ids_norm[0]]

def _select_close_candidates(candidates: List[RetrievedAnswer]) -> List[RetrievedAnswer]:
    if len(candidates) < 2:
        return []

    top = candidates[0]
    if top.score_final < SCORE_THRESHOLD:
        return []

    top_text_norm = normalize(top.answer.text)
    min_close_score = top.score_final * MULTI_ANSWER_RELATIVE_RATIO

    close_candidates: List[RetrievedAnswer] = []
    for candidate in candidates:
        if candidate.score_final < SCORE_THRESHOLD:
            break

        same_text = normalize(candidate.answer.text) == top_text_norm
        close_by_ratio = candidate.score_final >= (min_close_score - 1e-9)
        close_by_gap = (top.score_final - candidate.score_final) <= (MULTI_ANSWER_GAP_THRESHOLD + 1e-9)

        if same_text or close_by_ratio or close_by_gap:
            close_candidates.append(candidate)
            continue
        break

    if len(close_candidates) < 2:
        return []

    if len(close_candidates) > MAX_MULTI_ANSWER_CANDIDATES:
        return close_candidates

    next_idx = len(close_candidates)
    if next_idx < len(candidates):
        next_candidate = candidates[next_idx]
        next_far_enough = next_candidate.score_final <= (
            (top.score_final * MULTI_ANSWER_NEXT_FAR_RATIO) + 1e-9
        )
        if not next_far_enough:
            return []

    return close_candidates


def _format_multiple_answers(candidates: List[RetrievedAnswer]) -> str:
    intro = (
        f"Возможны {len(candidates)} релевантных варианта ответа по вашему запросу. "
        "Пожалуйста, используйте тот, который точнее соответствует вашему контексту.\n\n"
    )
    parts = [intro]
    for idx, candidate in enumerate(candidates, start=1):
        parts.append(f"Ответ {idx}:\n{candidate.answer.text}")
        if idx != len(candidates):
            parts.append("\n\n")
    return "".join(parts)


def _confirm_no_answer_with_llm(question: str) -> str:
    system_prompt = (
        "Ты проверяешь, можно ли безопасно вернуть пользователю сообщение об отсутствии точного ответа. "
        "Если retriever не нашел релевантных фрагментов базы знаний, а точного ответа действительно нет, "
        f"верни строго этот текст: {NO_ANSWER_TEXT} "
        "Не добавляй ничего от себя."
    )
    user_prompt = (
        f"Вопрос пользователя:\n{question}\n\n"
        "Retriever не нашел достаточно релевантных фрагментов базы знаний или признал найденные фрагменты "
        "нерелевантными. Подтверди, что безопасно вернуть стандартное сообщение об отсутствии точного ответа."
    )

    answer_text = call_llm_v2(system_prompt, user_prompt)
    if not answer_text or answer_text.startswith("Ошибка"):
        return MODEL_UNAVAILABLE_TEXT
    return NO_ANSWER_TEXT


def _return_no_answer(question: str, candidates: List[RetrievedAnswer]):
    return _confirm_no_answer_with_llm(question), None, candidates


def _return_service_unavailable(candidates: List[RetrievedAnswer]):
    return MODEL_UNAVAILABLE_TEXT, None, candidates


def answer_question_with_rag_v2(question: str):
    # Safety-net: даже при прямом вызове этой функции точные совпадения
    # возвращаем строго из базы и не отправляем в LLM.
    exact_answer = _match_exact_raw_question(question)
    if exact_answer is not None:
        return exact_answer.text, [exact_answer.id], []

    try:
        candidates = retrieve_answers(question, top_k=5)
    except Exception:
        # Недоступность retrieval/Qdrant не должна маскироваться под "ответ не найден".
        return _return_service_unavailable([])

    if not candidates:
        return _return_no_answer(question, candidates)

    top = candidates[0]
    if top.score_final >= (1.0 - 1e-9):
        return top.answer.text, [top.answer.id], candidates

    covered_candidates = [c for c in candidates if _candidate_has_query_coverage(question, c)]

    # "Ответ не найден" — только для явно нерелевантных запросов.
    # Запрос считаем нерелевантным, если одновременно:
    # 1) не нашлось ни одного кандидата с покрытием ключевых токенов вопроса;
    # 2) raw dense cosine у top-кандидата ниже LOW_RELEVANCE_DENSE_THRESHOLD;
    # 3) лексическое пересечение с вариантами вопроса ниже LOW_RELEVANCE_OVERLAP_THRESHOLD.
    top_overlap = _best_variant_overlap(question, top)
    if (
        not covered_candidates
        and top.score_dense_raw < LOW_RELEVANCE_DENSE_THRESHOLD
        and top_overlap < LOW_RELEVANCE_OVERLAP_THRESHOLD
    ):
        return _return_no_answer(question, candidates)

    ranked_candidates = covered_candidates if covered_candidates else candidates
    close_candidates = _select_close_candidates(ranked_candidates)
    if len(close_candidates) > MAX_MULTI_ANSWER_CANDIDATES:
        return TOO_MANY_RELEVANT_ANSWERS_TEXT, None, candidates
    if 2 <= len(close_candidates) <= MAX_MULTI_ANSWER_CANDIDATES:
        return _format_multiple_answers(close_candidates), [c.answer.id for c in close_candidates], candidates

    selected_candidate = ranked_candidates[0]

    # Для неточного совпадения используем RAG+LLM по одному лучшему выбранному ответу.
    # Передаем полный текст ответа (ans.text без обрезки) и ограниченный список вариантов вопросов.
    context_text = build_context_fragments([selected_candidate])

    system_prompt = (
        "Ты — ассистент по вопросам ОРД-А, ЕРИР и интернет-рекламы. "
        "Отвечай только по переданным фрагментам базы знаний. "
        "Нельзя придумывать факты. "
        "Если фрагмент содержит условия, ограничения, исключения, ссылки и списки — "
        "передай их полностью, без сокращений. "
        "Если точного ответа нет, напиши: \"В нашей базе знаний нет точного ответа на этот вопрос.\""
    )

    user_prompt = (
        f"Вопрос пользователя:\n{question}\n\n"
        f"Фрагменты базы знаний:\n{context_text}\n\n"
        "Сформулируй один точный ответ. Если один фрагмент полностью отвечает на вопрос, "
        "допустимо использовать максимально близкую к нему формулировку."
    )

    answer_text = call_llm_v2(system_prompt, user_prompt)
    if not answer_text or answer_text.startswith("Ошибка"):
        # Fallback: если LLM недоступна, всё равно отдаем детерминированный ответ из БЗ.
        return selected_candidate.answer.text, [selected_candidate.answer.id], candidates

    return answer_text, [selected_candidate.answer.id], candidates


# ---------- Итоговая логика: FAQ-словарь + RAG ----------

def answer_question_logic_v2(question: str):
    """
    1) Только 100% RAW-совпадение с вопросом из БЗ
       отдаем как готовый FAQ-ответ (без модели).
    2) Во всех остальных случаях используем RAG (retrieval + LLM).
    3) "Ответ не найден" возвращаем только для явно нерелевантных запросов.
    """
    raw_q = question.strip()

    # 1) 100% совпадение (raw): ответ строго из БЗ, без retrieval/LLM.
    exact_raw_answer = _match_exact_raw_question(question)
    if exact_raw_answer is not None:
        return {
            "answer": exact_raw_answer.text,
            "answer_source": "faq_exact",
            "answer_id": exact_raw_answer.id,
            "debug": {
                "matched_question_raw": raw_q,
                "matched_question_norm": None,
                "matched_answer_id": exact_raw_answer.id,
            },
        }

    # 2) RAG
    rag_answer, rag_answer_ids, candidates = answer_question_with_rag_v2(question)

    debug_candidates = [
        {
            "answer_id": c.answer.id,
            "score_dense": c.score_dense,
            "score_sparse": c.score_sparse,
            "score_dense_normalized": c.score_dense,
            "score_sparse_normalized": c.score_sparse,
            "score_final_normalized": c.score_final,
            "score_dense_raw": c.score_dense_raw,
            "score_sparse_raw": c.score_sparse_raw,
            "score_final": c.score_final,
            "categories": c.answer.categories,
        }
        for c in candidates
    ]

    source = "rag_llm"
    if not rag_answer_ids and rag_answer == NO_ANSWER_TEXT:
        source = "no_answer"

    return {
        "answer": rag_answer,
        "answer_source": source,
        "answer_id": rag_answer_ids[0] if rag_answer_ids else None,
        "debug": {"rag_answer_ids": rag_answer_ids},
        "debug_candidates": debug_candidates,
    }


# ---------- LEGACY V1 (без изменений логики) ----------

def retrieve_answers_v1(
    question: str,
    top_k: int = 5,
    dense_limit: int = 20,
    sparse_limit: int = 20,
) -> List[RetrievedAnswer]:
    norm_q = normalize(question)

    dense_q = dense_embed_query(question)

    expanded_q = expand_with_synonyms(norm_q)
    q_indices, q_values = compute_sparse_vector(expanded_q, TERM2ID, IDF)

    query_filter = models.Filter(
        must=[models.FieldCondition(key="is_active", match=models.MatchValue(value=True))]
    )

    dense_points = client_qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=dense_q,
        using="dense",
        query_filter=query_filter,
        limit=dense_limit,
        with_payload=True,
    ).points

    sparse_points = []
    if q_indices:
        sparse_vec = models.SparseVector(indices=q_indices, values=q_values)
        sparse_points = client_qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=sparse_vec,
            using="sparse",
            query_filter=query_filter,
            limit=sparse_limit,
            with_payload=True,
        ).points

    dense_scores: Dict[int, float] = {}
    sparse_scores: Dict[int, float] = {}

    for p in dense_points:
        dense_scores[int(p.id)] = float(p.score)

    for p in sparse_points:
        sparse_scores[int(p.id)] = float(p.score)

    def normalize_scores(scores: Dict[int, float]) -> Dict[int, float]:
        if not scores:
            return {}
        mx = max(scores.values())
        if mx <= 0:
            return {k: 0.0 for k, v in scores.items()}
        return {k: v / mx for k, v in scores.items()}

    # Важно: после такой нормализации лучший dense/sparse кандидат внутри текущей выборки
    # получает score 1.0, поэтому 1.0 здесь означает "лучший в shortlist", а не "точное совпадение".
    dense_norm = normalize_scores(dense_scores)
    sparse_norm = normalize_scores(sparse_scores)

    all_ids = set(dense_norm.keys()) | set(sparse_norm.keys())
    retrieved: List[RetrievedAnswer] = []
    src_points = {int(p.id): p for p in dense_points}
    src_points.update({int(p.id): p for p in sparse_points})

    for ans_id in all_ids:
        sd = dense_norm.get(ans_id, 0.0)
        ss = sparse_norm.get(ans_id, 0.0)
        sd_raw = dense_scores.get(ans_id, 0.0)
        ss_raw = sparse_scores.get(ans_id, 0.0)
        final = ALPHA_DENSE * sd + (1.0 - ALPHA_DENSE) * ss

        payload = src_points[ans_id].payload or {}
        ans = ANSWERS_BY_ID.get(ans_id)
        if not ans:
            ans = Answer(
                id=ans_id,
                text=payload.get("answer_text", ""),
                question_variants=payload.get("question_variants", []),
                categories=payload.get("categories", []),
                is_active=payload.get("is_active", True),
            )

        retrieved.append(
            RetrievedAnswer(
                answer=ans,
                score_dense=sd,
                score_sparse=ss,
                score_final=final,
                payload=payload,
            )
        )

    retrieved.sort(key=lambda r: r.score_final, reverse=True)
    return retrieved[:top_k]


def call_llm(system_prompt: str, user_prompt: str) -> str:
    if not LOCAL_LLM_URL:
        return "Ошибка: не настроен LOCAL_LLM_URL для локальной модели."

    full_prompt = system_prompt + "\n\n" + user_prompt

    headers = {
        "Content-Type": "application/json",
        "x-api-key": LOCAL_LLM_API_KEY,
    }
    payload = {
        "disable_reasoning": True,
        "messages": [
            {"role": "user", "content": full_prompt}
        ],
        "max_tokens": 640,
    }

    try:
        resp = requests.post(
            LOCAL_LLM_URL,
            json=payload,
            headers=headers,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data.get("content")
        if not isinstance(content, str):
            return "Ошибка: локальная модель вернула неожиданный формат ответа."
        return content.strip()
    except Exception as e:
        return f"Ошибка при обращении к локальной модели: {e}"


def answer_question_with_rag(question: str):
    try:
        candidates = retrieve_answers_v1(question, top_k=5)
    except Exception:
        # Не роняем legacy endpoint 500 при недоступном Qdrant/сети.
        return NO_ANSWER_TEXT, None, []

    if not candidates or candidates[0].score_final < SCORE_THRESHOLD_V1:
        system_prompt = (
            "Ты — ассистент по вопросам ОРД-А, ЕРИР и интернет-рекламы. "
            "Тебе не переданы релевантные фрагменты базы знаний. "
            "Ты обязан честно ответить, что точного ответа нет, "
            "и предложить обратиться к специалисту или переформулировать вопрос. "
            "Не выдумывай никаких фактов."
        )
        user_prompt = f"Вопрос пользователя:\n{question}"
        answer_text = call_llm(system_prompt, user_prompt)
        return answer_text, None, candidates

    context_text = build_context_fragments(candidates)

    system_prompt = (
        "Ты — ассистент по вопросам ОРД-А, ЕРИР и интернет-рекламы. "
        "Тебе передаются фрагменты базы знаний. "
        "Отвечай ТОЛЬКО на основе этих фрагментов.\n"
        "Если точного ответа в фрагментах нет, честно напиши: "
        "\"В нашей базе знаний нет точного ответа на этот вопрос.\" "
        "Не выдумывай новые факты, нормы закона, номера статей или ссылки.\n"
        "Отвечай кратко и по делу, на русском языке."
    )

    user_prompt = (
        f"Вопрос пользователя:\n{question}\n\n"
        f"Фрагменты базы знаний:\n{context_text}\n\n"
        "Сформулируй единый, понятный ответ для пользователя."
    )

    answer_text = call_llm(system_prompt, user_prompt)
    top_answer_id = candidates[0].answer.id if candidates else None
    return answer_text, top_answer_id, candidates


def answer_question_logic(question: str):
    raw_q = question.strip()

    direct_ids_raw = QUESTION_INDEX_RAW.get(raw_q)
    if direct_ids_raw:
        ans_id = direct_ids_raw[0]
        ans = ANSWERS_BY_ID[ans_id]
        return {
            "answer": ans.text,
            "answer_source": "faq_exact",
            "answer_id": ans.id,
            "debug": {
                "matched_question_raw": raw_q,
                "matched_question_norm": None,
                "matched_answer_id": ans.id,
            },
        }

    norm_q = normalize(question)
    direct_ids_norm = QUESTION_INDEX_NORMALIZED.get(norm_q)
    if direct_ids_norm:
        ans_id = direct_ids_norm[0]
        ans = ANSWERS_BY_ID[ans_id]
        return {
            "answer": ans.text,
            "answer_source": "faq_exact",
            "answer_id": ans.id,
            "debug": {
                "matched_question_raw": None,
                "matched_question_norm": norm_q,
                "matched_answer_id": ans.id,
            },
        }

    rag_answer, rag_answer_id, candidates = answer_question_with_rag(question)

    debug_candidates = [
        {
            "answer_id": c.answer.id,
            "score_dense": c.score_dense,
            "score_sparse": c.score_sparse,
            "score_dense_normalized": c.score_dense,
            "score_sparse_normalized": c.score_sparse,
            "score_final_normalized": c.score_final,
            "score_final": c.score_final,
            "categories": c.answer.categories,
        }
        for c in candidates
    ]

    source = "rag_llm"
    if rag_answer_id is None and (not candidates or candidates[0].score_final < SCORE_THRESHOLD_V1):
        source = "no_answer"

    return {
        "answer": rag_answer,
        "answer_source": source,
        "answer_id": rag_answer_id,
        "debug": {},
        "debug_candidates": debug_candidates,
    }


# ---------- FastAPI ----------

app = FastAPI(title="FAQ RAG Assistant")


@app.get("/health")
def health():
    return {"status": "ok"}


# 🔹 Простой эндпоинт — только текст ответа (JSON-строка)
@app.post("/answer", response_model=str, dependencies=[Depends(verify_api_key)])
def answer_simple(req: QueryRequest):
    """
    Возвращает только текст ответа (значение answer),
    без служебных полей и debug-информации.
    """
    result = answer_question_logic(req.question)
    return result["answer"]


# 🔹 Полный эндпоинт — вся структура (для отладки/аналитики)
@app.post("/answer_full", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
def answer_full(req: QueryRequest):
    """
    Возвращает полную структуру: answer, answer_source, answer_id и debug.
    """
    result = answer_question_logic(req.question)

    debug: Optional[Dict] = None
    if req.debug:
        dbg = result.get("debug", {}) or {}
        debug = {
            "matched_question_raw": dbg.get("matched_question_raw"),
            "matched_question_norm": dbg.get("matched_question_norm"),
            "matched_answer_id": dbg.get("matched_answer_id"),
            "rag_answer_ids": dbg.get("rag_answer_ids"),
            "candidates": result.get("debug_candidates"),
        }

    return QueryResponse(
        answer=result["answer"],
        answer_source=result["answer_source"],
        answer_id=result["answer_id"],
        debug=debug,
    )


@app.post("/answer_v2", response_model=str, dependencies=[Depends(verify_api_key)])
def answer_simple_v2(req: QueryRequest):
    result = answer_question_logic_v2(req.question)
    return result["answer"]


@app.post("/answer_full_v2", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
def answer_full_v2(req: QueryRequest):
    result = answer_question_logic_v2(req.question)

    debug: Optional[Dict] = None
    if req.debug:
        dbg = result.get("debug", {}) or {}
        debug = {
            "matched_question_raw": dbg.get("matched_question_raw"),
            "matched_question_norm": dbg.get("matched_question_norm"),
            "matched_answer_id": dbg.get("matched_answer_id"),
            "rag_answer_ids": dbg.get("rag_answer_ids"),
            "candidates": result.get("debug_candidates"),
        }

    return QueryResponse(
        answer=result["answer"],
        answer_source=result["answer_source"],
        answer_id=result["answer_id"],
        debug=debug,
    )
