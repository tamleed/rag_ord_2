import os
from typing import List

from qdrant_client import QdrantClient, models

from rag_core import (
    Answer,
    load_answers,
    build_tfidf_vocab,
    save_tfidf_vocab,
    normalize,
    expand_with_synonyms,
    compute_sparse_vector,
    dense_embed_passages,
    EMBED_DIM,
)

COLLECTION_NAME = "faq"
QA_DB_PATH = os.getenv("QA_DB_PATH", "./qa_db.json")
TFIDF_VOCAB_PATH = os.getenv("TFIDF_VOCAB_PATH", "./tfidf_vocab.json")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")


def ensure_faq_collection(client: QdrantClient) -> None:
    """
    Создаём коллекцию, только если её нет.
    Если есть — переиспользуем (upsert по тем же id перезапишет точки).
    """
    try:
        exists = client.collection_exists(collection_name=COLLECTION_NAME)
    except Exception as e:
        print(f"Не удалось проверить наличие коллекции: {e}")
        exists = False

    if exists:
        print(f"Collection '{COLLECTION_NAME}' already exists, will reuse it.")
        return

    print(f"Collection '{COLLECTION_NAME}' does not exist, creating...")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(
                size=EMBED_DIM,
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(),
        },
    )
    print(f"Collection '{COLLECTION_NAME}' created.")


def index_faq() -> None:
    # 1. Загружаем ответы
    answers: List[Answer] = load_answers(QA_DB_PATH)
    print(f"Loaded {len(answers)} answers from {QA_DB_PATH}")

    if not answers:
        print("No answers loaded, aborting.")
        return

    # 2. Строим TF-IDF словарь и сохраняем его
    term2id, idf = build_tfidf_vocab(answers)
    save_tfidf_vocab(TFIDF_VOCAB_PATH, term2id, idf)
    print(f"Saved TF-IDF vocab to {TFIDF_VOCAB_PATH} (terms: {len(term2id)})")

    # 3. Собираем базовый текст для эмбеддингов
    base_texts: List[str] = []
    for ans in answers:
        base = " ".join([ans.text] + ans.question_variants + ans.categories)
        base_texts.append(base)

    # 4. Считаем dense-эмбеддинги через BGE-M3
    dense_vectors = dense_embed_passages(base_texts)
    if len(dense_vectors) != len(answers):
        raise RuntimeError("Number of dense vectors does not match number of answers")

    # 5. Подключаемся к Qdrant с увеличенным таймаутом
    client = QdrantClient(
        url=QDRANT_URL,
        timeout=60.0,  # секунд на операции
    )

    # 6. Создаём коллекцию, если её нет
    ensure_faq_collection(client)

    # 7. Формируем точки для upsert (dense + sparse + payload)
    points: List[models.PointStruct] = []

    for ans, dense_vec, base_text in zip(answers, dense_vectors, base_texts):
        norm = normalize(base_text)
        expanded = expand_with_synonyms(norm)
        indices, values = compute_sparse_vector(expanded, term2id, idf)

        sparse_vec = models.SparseVector(
            indices=indices,
            values=values,
        )

        payload = {
            "answer_id": ans.id,
            "answer_text": ans.text,
            "question_variants": ans.question_variants,
            "categories": ans.categories,
            "is_active": ans.is_active,
        }

        point = models.PointStruct(
            id=ans.id,
            vector={
                "dense": dense_vec,
                "sparse": sparse_vec,
            },
            payload=payload,
        )
        points.append(point)

    # 8. Upsert всех точек (обновим / добавим)
    print(f"Upserting {len(points)} points into collection '{COLLECTION_NAME}'...")
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )
    print(f"Indexed {len(points)} answers into collection '{COLLECTION_NAME}'")


if __name__ == "__main__":
    index_faq()
