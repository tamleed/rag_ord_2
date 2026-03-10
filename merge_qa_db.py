#!/usr/bin/env python3
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

BASE_DIR = Path("/home/ubuntu/faq_rag")

# Укажем все файлы, которые хотим слить в мастер.
# Подстрой под свои реальные имена:
SOURCE_FILES = [
    BASE_DIR / "qa_db.json",         # старая база (если нужна)
    BASE_DIR / "qa_db2.json",        # вторая часть
    BASE_DIR / "qa_db_p3.json",      # третья часть / свежий дамп
    BASE_DIR / "qa_db_merged.json",  # текущий мастер (до обновления)
]

MASTER_FILE = BASE_DIR / "qa_db_merged.json"  # итоговый файл


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        # Формат типа: "2025-11-01T14:39:49.000000Z"
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        print(f"[WARN] Файл {path} не найден, пропускаю.")
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} должен содержать JSON-массив (список объектов)")
    print(f"[INFO] Загружено {len(data)} записей из {path.name}")
    return data


def normalize_q_text(text: str) -> str:
    """Простая нормализация текста вопроса для ключа."""
    return " ".join(text.strip().lower().split())


def merge_answers_with_questions(
    all_records: Dict[int, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """
    all_records[answer_id] -> список всех версий ответа из разных файлов.
    Объединяем:
      - сам ответ выбираем по максимальному updated_at;
      - вопросы объединяем по нормализованному тексту (union).
    """
    merged_answers: List[Dict[str, Any]] = []

    for aid, versions in sorted(all_records.items(), key=lambda x: x[0]):
        # 1) выбираем базовую версию ответа по updated_at
        best = None
        best_dt = None

        for rec in versions:
            dt = parse_dt(rec.get("updated_at"))
            if best is None:
                best = rec
                best_dt = dt
                continue
            if best_dt is None and dt is not None:
                best = rec
                best_dt = dt
                continue
            if dt is not None and best_dt is not None and dt >= best_dt:
                best = rec
                best_dt = dt

        if best is None:
            # теоретически не должно быть, но на всякий случай
            continue

        # базовая копия ответа
        base = dict(best)
        base["id"] = aid

        # 2) объединяем категории на уровне ответа
        cats = set()
        for rec in versions:
            rc = rec.get("category") or rec.get("categories") or []
            if isinstance(rc, str):
                cats.add(rc)
            else:
                cats.update(rc)
        base["category"] = sorted(cats)

        # 3) объединяем вопросы по нормализованному тексту
        q_by_text: Dict[str, Dict[str, Any]] = {}

        for rec in versions:
            questions = rec.get("questions") or []
            for q in questions:
                q_text = q.get("text")
                if not q_text:
                    continue
                key = normalize_q_text(q_text)

                existing = q_by_text.get(key)
                if existing is None:
                    q_by_text[key] = dict(q)
                    continue

                # если вопрос уже есть – решаем по updated_at и is_active
                old_dt = parse_dt(existing.get("updated_at"))
                new_dt = parse_dt(q.get("updated_at"))

                # выбираем более новый updated_at
                take_new = False
                if old_dt is None and new_dt is not None:
                    take_new = True
                elif old_dt is not None and new_dt is not None and new_dt >= old_dt:
                    take_new = True

                if take_new:
                    q_by_text[key] = dict(q)

                # если хоть в одной версии вопрос is_active=True, считаем его активным
                if existing.get("is_active", True) is False and q.get("is_active", True) is True:
                    q_by_text[key]["is_active"] = True

        # 4) финализируем список вопросов
        merged_questions = []
        for key, q in q_by_text.items():
            q = dict(q)
            # гарантируем правильный answer_id
            q["answer_id"] = aid
            merged_questions.append(q)

        # можно отсортировать вопросы по тексту/updated_at для стабильности
        merged_questions.sort(key=lambda x: normalize_q_text(x.get("text", "")))

        base["questions"] = merged_questions

        merged_answers.append(base)

    return merged_answers


def main():
    # Собираем все версии ответов по id
    all_records: Dict[int, List[Dict[str, Any]]] = {}

    for path in SOURCE_FILES:
        records = load_json_list(path)
        for item in records:
            aid = item.get("id")
            if aid is None:
                continue
            all_records.setdefault(aid, []).append(item)

    print(f"[INFO] Всего уникальных answer.id: {len(all_records)}")

    # Объединяем ответы + вопросы
    merged = merge_answers_with_questions(all_records)
    print(f"[INFO] Ответов после merge: {len(merged)}")

    # Бэкапим старый master-файл
    if MASTER_FILE.exists():
        backup = MASTER_FILE.with_suffix(".json.bak")
        MASTER_FILE.replace(backup)
        print(f"[INFO] Старый {MASTER_FILE.name} сохранён как {backup.name}")

    with MASTER_FILE.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Новый мастер-файл сохранён в {MASTER_FILE}")


if __name__ == "__main__":
    main()
