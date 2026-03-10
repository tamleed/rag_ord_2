import os
from typing import List, Dict, Optional

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
COLLECTION_NAME = "faq"

ALPHA_DENSE = 0.6          # вес dense-вектора
SCORE_THRESHOLD = 0.2      # порог уверенности retriever'а

# локальная LLM
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "")
LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "")
  # API-ключ для доступа к самому RAG-API
RAG_API_KEY = os.getenv("RAG_API_KEY", "")


# ---------- Глобальное состояние ----------

client_qdrant: QdrantClient = QdrantClient(url=QDRANT_URL)

ANSWERS: List[Answer] = load_answers(QA_DB_PATH)
ANSWERS_BY_ID: Dict[int, Answer] = {a.id: a for a in ANSWERS}

QUESTION_INDEX_RAW: Dict[str, List[int]] = build_raw_q_index(ANSWERS)
QUESTION_INDEX_NORMALIZED: Dict[str, List[int]] = build_normalized_q_index(ANSWERS)

TERM2ID, IDF = load_tfidf_vocab(TFIDF_VOCAB_PATH)


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
    ):
        self.answer = answer
        self.score_dense = score_dense
        self.score_sparse = score_sparse
        self.score_final = score_final
        self.payload = payload


# ---------- Retriever (hybrid dense + sparse) ----------

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
    q_indices, q_values = compute_sparse_vector(expanded_q, TERM2ID, IDF)

    query_filter = models.Filter(
        must=[models.FieldCondition(key="is_active", match=models.MatchValue(value=True))]
    )

    # 3. dense-запрос
    dense_points = client_qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=dense_q,
        using="dense",
        query_filter=query_filter,
        limit=dense_limit,
        with_payload=True,
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

    dense_norm = normalize_scores(dense_scores)
    sparse_norm = normalize_scores(sparse_scores)

    all_ids = set(dense_norm.keys()) | set(sparse_norm.keys())
    retrieved: List[RetrievedAnswer] = []
    src_points = {int(p.id): p for p in dense_points}
    src_points.update({int(p.id): p for p in sparse_points})

    for ans_id in all_ids:
        sd = dense_norm.get(ans_id, 0.0)
        ss = sparse_norm.get(ans_id, 0.0)
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


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Вызываем локальную модель по HTTP.

    Формат:
    POST LOCAL_LLM_URL
      headers: x-api-key: LOCAL_LLM_API_KEY
      body: {
        "disable_reasoning": true,
        "messages": [{"role": "user", "content": "<full_prompt>"}],
        "max_tokens": 640
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
    candidates = retrieve_answers(question, top_k=5)

    if not candidates or candidates[0].score_final < SCORE_THRESHOLD:
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


# ---------- Итоговая логика: FAQ-словарь + RAG ----------

def answer_question_logic(question: str):
    """
    1) Классический режим: ищем ТОЛЬКО по вопросам (questions[].text).
       - сначала RAW-совпадение;
       - затем нормализованное.
       Если нашли — отдаём готовый Answer.text.
    2) Если нет — запускаем RAG (dense + sparse + локальная LLM).
    """
    raw_q = question.strip()

    # 1. Точный матч по "сырым" вопросам
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

    # 2. Матч по нормализованным вопросам
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

    # 3. RAG
    rag_answer, rag_answer_id, candidates = answer_question_with_rag(question)

    debug_candidates = [
        {
            "answer_id": c.answer.id,
            "score_dense": c.score_dense,
            "score_sparse": c.score_sparse,
            "score_final": c.score_final,
            "categories": c.answer.categories,
        }
        for c in candidates
    ]

    source = "rag_llm"
    if rag_answer_id is None and (not candidates or candidates[0].score_final < SCORE_THRESHOLD):
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
            "candidates": result.get("debug_candidates"),
        }

    return QueryResponse(
        answer=result["answer"],
        answer_source=result["answer_source"],
        answer_id=result["answer_id"],
        debug=debug,
    )
