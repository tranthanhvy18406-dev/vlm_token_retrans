import argparse
import hashlib
import heapq
import io
import json
import os
import sys
import zipfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts import _gqa_download_utils


def answer_allowed(answer: str, excluded_answers: set[str], max_answer_words: int) -> bool:
    normalized = answer.strip().lower()
    if not normalized:
        return False
    if normalized in excluded_answers:
        return False
    return len(normalized.split()) <= max_answer_words


def stable_score(key: str, seed: int) -> float:
    digest = hashlib.blake2b(f"{seed}:{key}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64 - 1)


def iter_json_object_members(zip_path: str, member: str, chunk_size: int = 1024 * 1024):
    """
    Stream a large top-level JSON object as (key, value) pairs.

    GQA question files are hundreds of MB when decompressed, and json.load can be
    killed on shared login nodes. This parser keeps only the current object in
    memory.
    """
    decoder = json.JSONDecoder()

    with zipfile.ZipFile(zip_path) as zf, zf.open(member) as raw:
        reader = io.TextIOWrapper(raw, encoding="utf-8")
        buffer = ""
        pos = 0
        eof = False

        def fill():
            nonlocal buffer, eof
            if eof:
                return
            chunk = reader.read(chunk_size)
            if chunk == "":
                eof = True
            else:
                buffer += chunk

        def compact():
            nonlocal buffer, pos
            if pos > chunk_size:
                buffer = buffer[pos:]
                pos = 0

        def skip_ws():
            nonlocal pos
            while True:
                while pos < len(buffer) and buffer[pos].isspace():
                    pos += 1
                if pos < len(buffer) or eof:
                    return
                fill()

        def expect_char(ch: str):
            nonlocal pos
            skip_ws()
            while pos >= len(buffer) and not eof:
                fill()
                skip_ws()
            if pos >= len(buffer) or buffer[pos] != ch:
                got = "<eof>" if pos >= len(buffer) else repr(buffer[pos])
                raise ValueError(f"Expected {ch!r} while parsing {member}, got {got}.")
            pos += 1

        def decode_next():
            nonlocal pos
            skip_ws()
            while True:
                try:
                    value, end = decoder.raw_decode(buffer, pos)
                    pos = end
                    compact()
                    return value
                except json.JSONDecodeError:
                    if eof:
                        raise
                    fill()

        fill()
        expect_char("{")

        while True:
            skip_ws()
            while pos >= len(buffer) and not eof:
                fill()
                skip_ws()
            if pos < len(buffer) and buffer[pos] == "}":
                pos += 1
                return

            key = decode_next()
            if not isinstance(key, str):
                raise ValueError(f"Expected object key string in {member}.")
            expect_char(":")
            value = decode_next()
            yield key, value

            skip_ws()
            while pos >= len(buffer) and not eof:
                fill()
                skip_ws()
            if pos < len(buffer) and buffer[pos] == ",":
                pos += 1
                continue
            if pos < len(buffer) and buffer[pos] == "}":
                pos += 1
                return
            got = "<eof>" if pos >= len(buffer) else repr(buffer[pos])
            raise ValueError(f"Expected ',' or '}}' while parsing {member}, got {got}.")


def collect_candidate_pool(
    questions_zip: str,
    member: str,
    split_name: str,
    count: int,
    seed: int,
    image_dir: str,
    used_qids: set[str],
    used_image_ids: set[str],
    excluded_answers: set[str],
    max_answer_words: int,
    disjoint_images: bool,
    candidate_pool_factor: int,
    max_scan_items: int | None,
) -> tuple[list[dict], dict[str, int]]:
    heap: list[tuple[float, int, dict]] = []
    pool_limit = max(count * candidate_pool_factor, count + 500)
    stats = {
        "scanned": 0,
        "balanced": 0,
        "filtered": 0,
        "preselected": 0,
        "skipped_used": 0,
    }

    for qid, q in iter_json_object_members(questions_zip, member):
        stats["scanned"] += 1
        if max_scan_items is not None and stats["scanned"] > max_scan_items:
            break
        if not q.get("isBalanced", False):
            continue
        stats["balanced"] += 1

        answer = str(q.get("answer", "")).strip()
        if not answer_allowed(answer, excluded_answers, max_answer_words):
            continue
        stats["filtered"] += 1

        image_id = str(q["imageId"])
        if qid in used_qids or (disjoint_images and image_id in used_image_ids):
            stats["skipped_used"] += 1
            continue

        score = stable_score(qid, seed)
        sample = {
            "image": os.path.join(image_dir, f"{image_id}.jpg"),
            "question": q["question"],
            "answer": answer,
            "qid": qid,
            "image_id": image_id,
            "source": f"gqa_{split_name}",
            "_score": score,
        }
        entry = (-score, stats["scanned"], sample)
        if len(heap) < pool_limit:
            heapq.heappush(heap, entry)
            stats["preselected"] += 1
        elif score < -heap[0][0]:
            heapq.heapreplace(heap, entry)

    candidates = [entry[2] for entry in heap]
    candidates.sort(key=lambda item: item["_score"])
    return candidates, stats


def build_split(
    questions_zip: str,
    member: str,
    split_name: str,
    count: int,
    seed: int,
    image_dir: str,
    used_qids: set[str],
    used_image_ids: set[str],
    excluded_answers: set[str],
    max_answer_words: int,
    disjoint_images: bool,
    candidate_pool_factor: int,
    max_scan_items: int | None,
) -> tuple[list[dict], dict[str, int]]:
    candidates, stats = collect_candidate_pool(
        questions_zip=questions_zip,
        member=member,
        split_name=split_name,
        count=count,
        seed=seed,
        image_dir=image_dir,
        used_qids=used_qids,
        used_image_ids=used_image_ids,
        excluded_answers=excluded_answers,
        max_answer_words=max_answer_words,
        disjoint_images=disjoint_images,
        candidate_pool_factor=candidate_pool_factor,
        max_scan_items=max_scan_items,
    )

    samples = []
    local_image_ids: set[str] = set()
    download_failed = 0
    duplicate_image = 0

    for sample in candidates:
        image_id = sample["image_id"]
        if disjoint_images and (image_id in used_image_ids or image_id in local_image_ids):
            duplicate_image += 1
            continue

        if not _gqa_download_utils.download_vg_image(image_id, sample["image"]):
            download_failed += 1
            continue

        clean_sample = {key: value for key, value in sample.items() if key != "_score"}
        samples.append(clean_sample)
        used_qids.add(sample["qid"])
        used_image_ids.add(image_id)
        local_image_ids.add(image_id)
        if len(samples) >= count:
            break

    stats["download_failed"] = download_failed
    stats["duplicate_image"] = duplicate_image
    stats["written"] = len(samples)

    if len(samples) < count:
        raise RuntimeError(
            f"Only wrote {len(samples)} / {count} samples for split {split_name}; "
            "increase --candidate_pool_factor, relax filters, or disable disjoint_images."
        )
    return samples, stats


def write_jsonl(path: str, samples: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=True) + "\n")


def print_zip_member_info(zip_path: str, members: list[str]) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        for member in members:
            info = zf.getinfo(member)
            size_mib = info.file_size / 1024**2
            print(f"{member}: {size_mib:.1f} MiB decompressed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--image_dir", default="data/gqa_images")
    parser.add_argument("--train_output", default="data/gqa_train1000.jsonl")
    parser.add_argument("--test_output", default="data/gqa_test300.jsonl")
    parser.add_argument("--train_count", type=int, default=1000)
    parser.add_argument("--test_count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include_yesno", action="store_true")
    parser.add_argument("--max_answer_words", type=int, default=3)
    parser.add_argument("--allow_image_overlap", action="store_true")
    parser.add_argument("--candidate_pool_factor", type=int, default=50)
    parser.add_argument("--max_scan_items", type=int, default=None)
    args = parser.parse_args()

    members = ["train_balanced_questions.json", "val_balanced_questions.json"]
    print_zip_member_info(args.questions, members)

    excluded = set() if args.include_yesno else {"yes", "no"}
    disjoint_images = not args.allow_image_overlap

    used_qids: set[str] = set()
    used_image_ids: set[str] = set()

    train_samples, train_stats = build_split(
        questions_zip=args.questions,
        member="train_balanced_questions.json",
        split_name="train",
        count=args.train_count,
        seed=args.seed,
        image_dir=args.image_dir,
        used_qids=used_qids,
        used_image_ids=used_image_ids,
        excluded_answers=excluded,
        max_answer_words=args.max_answer_words,
        disjoint_images=disjoint_images,
        candidate_pool_factor=args.candidate_pool_factor,
        max_scan_items=args.max_scan_items,
    )
    test_samples, test_stats = build_split(
        questions_zip=args.questions,
        member="val_balanced_questions.json",
        split_name="test",
        count=args.test_count,
        seed=args.seed + 1,
        image_dir=args.image_dir,
        used_qids=used_qids,
        used_image_ids=used_image_ids,
        excluded_answers=excluded,
        max_answer_words=args.max_answer_words,
        disjoint_images=disjoint_images,
        candidate_pool_factor=args.candidate_pool_factor,
        max_scan_items=args.max_scan_items,
    )

    write_jsonl(args.train_output, train_samples)
    write_jsonl(args.test_output, test_samples)

    print(f"wrote train: {len(train_samples)} to {args.train_output}, stats={train_stats}")
    print(f"wrote test: {len(test_samples)} to {args.test_output}, stats={test_stats}")
    print(f"unique image_ids across splits: {len(used_image_ids)}")


if __name__ == "__main__":
    main()
