import json
import math
import re
from dataclasses import dataclass
from collections import Counter
from typing import List, Dict, Tuple, Optional

from sentence_transformers import SentenceTransformer


# ---------- Модель данных ----------

@dataclass
class Answer:
    id: int
    text: str
    question_variants: List[str]
    categories: List[str]
    is_active: bool


def load_answers(path: str) -> List[Answer]:
    """Загружаем qa_db.json и нормализуем в список Answer."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    answers: List[Answer] = []

    for item in raw:
        ans_id = item["id"]
        is_active = bool(item.get("is_active", True))
        text = item["text"]

        # категории могут быть в category или categories
        cats = item.get("categories") or item.get("category") or []
        if isinstance(cats, str):
            cats = [cats]

        q_variants: List[str] = []
        for q in item.get("questions", []):
            if not q.get("is_active", True):
                continue
            q_text = q["text"]
            q_variants.append(q_text)

        answers.append(
            Answer(
                id=ans_id,
                text=text,
                question_variants=q_variants,
                categories=cats,
                is_active=is_active,
            )
        )
    return answers


# ---------- Нормализация / токенизация / синонимы ----------

PUNCT_RE = re.compile(r"[^\w\s%/]", flags=re.UNICODE)
SPACE_RE = re.compile(r"\s+", flags=re.UNICODE)


def normalize(text: str) -> str:
    """
    Простая нормализация под русскоязычный текст:
    - нижний регистр
    - ё -> е
    - дефис -> пробел (чтобы 'ОРД-А' стало 'орд а')
    - удаляем пунктуацию (кроме % и /)
    - сжимаем пробелы
    """
    text = text.lower()
    text = text.replace("ё", "е")
    text = text.replace("-", " ")
    text = PUNCT_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return [t for t in text.split() if t]


# Группы синонимов в нормализованном виде
SYNONYM_GROUPS: List[List[str]] = [
    # ЛК
    ["лк", "личный кабинет", "личный кабинет орд а", "кабинет орд а"],
    # ЕРИР
    [
        "ерир",
        "единый реестр интернет рекламы",
        "единый реестр интернет рекламы роскомнадзора",
        "реестр интернет рекламы",
    ],
    # ОРД-А
    ["орд а", "оператор рекламных данных", "оператор рекламных данных а"],
    # ERID / идентификаторы
    [
        "erid",
        "идентификатор рекламы",
        "идентификатор объявления",
        "идентификатор объекта",
        "ид объявления",
        "ид объекта",
    ],
    # ИД / id
    ["ид", "id", "идентификатор"],
    # РК
    ["рк", "рекламная кампания"],
    # Креатив / объявление / объект
    ["креатив", "объявление", "рекламный объект", "баннер", "ролик"],
    # Разаллокация — вариации/опечатки
    ["разаллокация", "разалокация", "разоллакация", "разалакация"],
]


def expand_with_synonyms(normalized_text: str) -> str:
    """
    Если в тексте есть термин из группы синонимов — добавляем все синонимы в "хвост".
    Так sparse-вектор и запрос, и документ будут знать про все варианты.
    """
    parts = [normalized_text]
    for group in SYNONYM_GROUPS:
        for term in group:
            if term in normalized_text:
                parts.append(" ".join(group))
                break
    return " ".join(parts)


# ---------- TF-IDF (sparse) ----------

def build_tfidf_vocab(answers: List[Answer]) -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    Строим:
    - term2id: str -> int
    - idf: str -> float
    """
    N = len(answers)
    df = Counter()

    for ans in answers:
        base_text = " ".join([ans.text] + ans.question_variants + ans.categories)
        norm = normalize(base_text)
        expanded = expand_with_synonyms(norm)
        tokens = tokenize(expanded)
        unique_terms = set(tokens)
        df.update(unique_terms)

    terms = sorted(df.keys())
    term2id: Dict[str, int] = {t: i for i, t in enumerate(terms)}

    idf: Dict[str, float] = {}
    for t in terms:
        idf[t] = math.log((N + 1) / (df[t] + 1)) + 1.0

    return term2id, idf


def compute_sparse_vector(
    expanded_normalized_text: str,
    term2id: Dict[str, int],
    idf: Dict[str, float],
) -> Tuple[List[int], List[float]]:
    tokens = tokenize(expanded_normalized_text)
    tf = Counter(tokens)

    indices: List[int] = []
    values: List[float] = []

    for term, freq in tf.items():
        if term not in term2id:
            continue
        idx = term2id[term]
        w = freq * idf.get(term, 0.0)
        if w == 0.0:
            continue
        indices.append(idx)
        values.append(float(w))

    return indices, values


def save_tfidf_vocab(path: str, term2id: Dict[str, int], idf: Dict[str, float]) -> None:
    data = {
        "term2id": term2id,
        "idf": idf,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def load_tfidf_vocab(path: str) -> Tuple[Dict[str, int], Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    term2id = {k: int(v) for k, v in data["term2id"].items()}
    idf = {k: float(v) for k, v in data["idf"].items()}
    return term2id, idf


# ---------- Эмбеддинги (dense) на BGE-M3 (CPU) ----------

EMBED_MODEL_NAME = "BAAI/bge-m3"
EMBED_DIM = 1024  # у bge-m3 размерность 1024

_embed_model: Optional[SentenceTransformer] = None


def get_embed_model() -> SentenceTransformer:
    """
    Загружаем BGE-M3 только на CPU.
    Если на машине есть GPU, всё равно можно форсировать CPU через device="cpu".
    """
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
    return _embed_model


def dense_embed_passages(texts: List[str]) -> List[List[float]]:
    """
    Эмбеддинги для документов (ответов).
    Для BGE-M3 достаточно кодировать текст как есть.
    """
    model = get_embed_model()
    embs = model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embs.tolist()


def dense_embed_query(text: str) -> List[float]:
    """
    Эмбеддинг для запроса.
    """
    model = get_embed_model()
    emb = model.encode(
        [text],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]
    return emb.tolist()


# ---------- Индексы по вопросам ----------

def build_normalized_q_index(answers: List[Answer]) -> Dict[str, List[int]]:
    """
    Индекс: нормализованный текст вопроса -> список id Answer.
    """
    idx: Dict[str, List[int]] = {}
    for ans in answers:
        for q in ans.question_variants:
            key = normalize(q)
            idx.setdefault(key, []).append(ans.id)
    return idx


def build_raw_q_index(answers: List[Answer]) -> Dict[str, List[int]]:
    """
    Классический индекс:
    Сырой текст вопроса (как в qa_db.json) -> список id Answer.
    Сравнение по точному совпадению строк.
    """
    idx: Dict[str, List[int]] = {}
    for ans in answers:
        for q in ans.question_variants:
            key = q.strip()
            idx.setdefault(key, []).append(ans.id)
    return idx
