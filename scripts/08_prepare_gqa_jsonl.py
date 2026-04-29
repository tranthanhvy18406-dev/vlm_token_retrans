import argparse
import json
import os
import random
import zipfile


def iter_question_members(zip_path: str):
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith(".json") and not name.endswith("/"):
                yield name


def pick_question_member(zip_path: str, split: str, balanced_only: bool) -> str:
    members = list(iter_question_members(zip_path))
    if not members:
        raise RuntimeError(f"No .json question files found in {zip_path}")

    def score(name: str) -> tuple[int, int, int]:
        base = os.path.basename(name).lower()
        split_score = 1 if split.lower() in base else 0
        balanced_score = 1 if "balanced" in base else 0
        all_score = 1 if "all" in base else 0
        if balanced_only:
            return split_score, balanced_score, -all_score
        return split_score, all_score, balanced_score

    best = max(members, key=score)
    if split.lower() not in os.path.basename(best).lower():
        raise RuntimeError(
            f"Could not find a question file for split={split!r}. Available: {members[:20]}"
        )
    if balanced_only and "balanced" not in os.path.basename(best).lower():
        raise RuntimeError(
            f"Could not find a balanced question file for split={split!r}. "
            f"Picked {best}; available: {members[:20]}"
        )
    return best


def load_questions_for_split(path: str, split: str, balanced_only: bool, member: str | None):
    if zipfile.is_zipfile(path):
        chosen_member = member or pick_question_member(path, split=split, balanced_only=balanced_only)
        with zipfile.ZipFile(path) as zf, zf.open(chosen_member) as handle:
            return json.load(handle), chosen_member
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle), path


def image_path_for(image_dir: str, image_id: str) -> str:
    return os.path.join(image_dir, f"{image_id}.jpg")


def build_image_zip_index(image_zip: str) -> dict[str, str]:
    index = {}
    with zipfile.ZipFile(image_zip) as zf:
        for name in zf.namelist():
            if name.endswith(".jpg") and not name.endswith("/"):
                index[os.path.basename(name)] = name
    return index


def extract_image(image_zf: zipfile.ZipFile, member: str, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        return
    with image_zf.open(member) as src, open(output_path, "wb") as dst:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)


def answer_allowed(answer: str, excluded_answers: set[str], max_answer_words: int) -> bool:
    normalized = answer.strip().lower()
    if not normalized:
        return False
    if normalized in excluded_answers:
        return False
    return len(normalized.split()) <= max_answer_words


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--question_member", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", default="data/gqa_gqa200.jsonl")
    parser.add_argument("--image_dir", default="")
    parser.add_argument("--image_zip", default="")
    parser.add_argument("--extract_images_dir", default="data/gqa_images")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include_unbalanced", action="store_true")
    parser.add_argument("--include_yesno", action="store_true")
    parser.add_argument("--max_answer_words", type=int, default=3)
    parser.add_argument("--allow_missing_images", action="store_true")
    args = parser.parse_args()

    balanced_only = not args.include_unbalanced
    questions, source = load_questions_for_split(
        path=args.questions,
        split=args.split,
        balanced_only=balanced_only,
        member=args.question_member,
    )
    print(f"loaded questions from {source}: {len(questions)}")

    image_zip_index = None
    image_zf = None
    if args.image_zip:
        image_zip_index = build_image_zip_index(args.image_zip)
        print(f"indexed image zip: {len(image_zip_index)} jpg files")
        image_zf = zipfile.ZipFile(args.image_zip)

    excluded = set() if args.include_yesno else {"yes", "no"}
    items = list(questions.items())
    rng = random.Random(args.seed)
    rng.shuffle(items)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    written = 0
    skipped = 0

    try:
        with open(args.output, "w", encoding="utf-8") as out:
            for qid, q in items:
                if balanced_only and not q.get("isBalanced", False):
                    skipped += 1
                    continue
                answer = str(q.get("answer", "")).strip()
                if not answer_allowed(answer, excluded, args.max_answer_words):
                    skipped += 1
                    continue

                image_id = str(q["imageId"])
                if args.image_zip:
                    image_path = image_path_for(args.extract_images_dir, image_id)
                    member = image_zip_index.get(f"{image_id}.jpg") if image_zip_index else None
                    if member is None:
                        skipped += 1
                        continue
                    extract_image(image_zf, member, image_path)
                elif args.image_dir:
                    image_path = image_path_for(args.image_dir, image_id)
                else:
                    image_path = f"{image_id}.jpg"

                if not args.allow_missing_images and not os.path.exists(image_path):
                    skipped += 1
                    continue

                sample = {
                    "image": image_path,
                    "question": q["question"],
                    "answer": answer,
                    "qid": qid,
                    "image_id": image_id,
                    "source": "gqa",
                }
                out.write(json.dumps(sample, ensure_ascii=True) + "\n")
                written += 1
                if written >= args.max_samples:
                    break
    finally:
        if image_zf is not None:
            image_zf.close()

    print(f"wrote {written} samples to {args.output}")
    print(f"skipped {skipped} questions")
    if written == 0:
        raise RuntimeError("No GQA samples were written; check filters and image paths.")


if __name__ == "__main__":
    main()
