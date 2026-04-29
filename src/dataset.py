import json
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class VQASample:
    image: str
    question: str
    answer: str


def load_jsonl(
    path: str,
    image_root: str = "",
    max_samples: Optional[int] = None,
) -> list[VQASample]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)

            for key in ["image", "question", "answer"]:
                if key not in obj:
                    raise KeyError(f"{path}:{line_no} missing required field {key!r}")

            image_path = obj["image"]
            if image_root and not os.path.isabs(image_path):
                image_path = os.path.join(image_root, image_path)

            samples.append(
                VQASample(
                    image=image_path,
                    question=obj["question"],
                    answer=str(obj["answer"]),
                )
            )

            if max_samples is not None and len(samples) >= max_samples:
                break

    if not samples:
        raise RuntimeError(f"No samples loaded from {path}")
    return samples
