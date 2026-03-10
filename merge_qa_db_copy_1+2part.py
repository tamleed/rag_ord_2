import json
import os
from typing import Dict, Any, List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# исходные файлы
BASE_FILE = os.path.join(BASE_DIR, "qa_db.json")
EXTRA_FILE = os.path.join(BASE_DIR, "qa_db2.json")
OUT_FILE = os.path.join(BASE_DIR, "qa_db_merged.json")


def load_list(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON array")
    return data


def main():
    base = load_list(BASE_FILE)
    extra = load_list(EXTRA_FILE)

    print(f"Base records:  {len(base)}")
    print(f"Extra records: {len(extra)}")

    # Соберём по id; если id совпадает, отдаём приоритет extra (новая база как патч)
    by_id: Dict[int, Dict[str, Any]] = {}

    for item in base:
        ans_id = int(item["id"])
        by_id[ans_id] = item

    for item in extra:
        ans_id = int(item["id"])
        # при совпадении id новое определение перезаписывает старое
        by_id[ans_id] = item

    merged: List[Dict[str, Any]] = []
    for ans_id in sorted(by_id.keys()):
        item = dict(by_id[ans_id])

        # гарантируем, что id целое и согласовано
        item["id"] = ans_id

        # корректируем answer_id внутри questions, если поле есть
        questions = item.get("questions") or []
        for q in questions:
            q["answer_id"] = ans_id

        merged.append(item)

    print(f"Merged records: {len(merged)}")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"Saved merged DB to {OUT_FILE}")


if __name__ == "__main__":
    main()
