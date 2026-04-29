import argparse
import os
import sys
import urllib.error
import urllib.request


GQA_URLS = {
    "questions": "https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip",
    "images": "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip",
    "scene_graphs": "https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip",
}


def remote_size(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        length = response.headers.get("Content-Length")
    return int(length) if length is not None else None


def download_with_resume(url: str, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    expected_size = remote_size(url)

    if expected_size is not None and os.path.exists(output_path):
        local_size = os.path.getsize(output_path)
        if local_size == expected_size:
            print(f"exists: {output_path} ({local_size} bytes)")
            return
        print(f"size mismatch, redownloading: {output_path} ({local_size} != {expected_size})")
        os.remove(output_path)

    part_path = output_path + ".part"
    start = os.path.getsize(part_path) if os.path.exists(part_path) else 0

    headers = {}
    mode = "ab"
    if start > 0:
        headers["Range"] = f"bytes={start}-"
        print(f"resuming: {output_path} from byte {start}")
    else:
        print(f"downloading: {url}")

    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code == 416:
            os.replace(part_path, output_path)
            return
        raise

    if start > 0 and response.status != 206:
        print("server did not honor Range; restarting download")
        start = 0
        mode = "wb"

    downloaded = start
    chunk_size = 1024 * 1024
    with response, open(part_path, mode) as handle:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if expected_size:
                pct = 100.0 * downloaded / expected_size
                print(f"\r{os.path.basename(output_path)} {pct:6.2f}% ", end="", flush=True)
    print()

    if expected_size is not None and os.path.getsize(part_path) != expected_size:
        raise RuntimeError(
            f"incomplete download for {output_path}: "
            f"{os.path.getsize(part_path)} != {expected_size}"
        )
    os.replace(part_path, output_path)
    print(f"saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/scratch/prj/nmes_simeone/datasets/gqa")
    parser.add_argument("--questions", action="store_true")
    parser.add_argument("--images", action="store_true")
    parser.add_argument("--scene_graphs", action="store_true")
    args = parser.parse_args()

    selected = []
    if args.questions:
        selected.append("questions")
    if args.images:
        selected.append("images")
    if args.scene_graphs:
        selected.append("scene_graphs")
    if not selected:
        selected.append("questions")

    filenames = {
        "questions": "questions1.2.zip",
        "images": "images.zip",
        "scene_graphs": "sceneGraphs.zip",
    }
    for key in selected:
        download_with_resume(GQA_URLS[key], os.path.join(args.root, filenames[key]))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
