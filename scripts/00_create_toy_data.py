import argparse
import json
import os

from PIL import Image, ImageDraw


COLORS = [
    ("red", (220, 40, 40)),
    ("blue", (40, 90, 220)),
    ("green", (40, 170, 80)),
    ("yellow", (235, 200, 40)),
    ("black", (20, 20, 20)),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--num-samples", type=int, default=20)
    args = parser.parse_args()

    image_dir = os.path.join(args.output_dir, "examples")
    os.makedirs(image_dir, exist_ok=True)
    jsonl_path = os.path.join(args.output_dir, "gqa_mini.jsonl")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for idx in range(args.num_samples):
            color_name, rgb = COLORS[idx % len(COLORS)]
            image = Image.new("RGB", (336, 336), (245, 245, 235))
            draw = ImageDraw.Draw(image)
            margin = 70 + (idx % 3) * 10
            draw.rectangle([margin, margin, 336 - margin, 336 - margin], fill=rgb)
            draw.ellipse([25, 25, 75, 75], fill=(120, 120, 120))

            rel_path = f"examples/{idx:04d}.jpg"
            image.save(os.path.join(args.output_dir, rel_path), quality=95)

            sample = {
                "image": os.path.join(args.output_dir, rel_path),
                "question": "What color is the large square?",
                "answer": color_name,
            }
            f.write(json.dumps(sample) + "\n")

    print(f"Wrote {args.num_samples} samples to {jsonl_path}")


if __name__ == "__main__":
    main()
