import os
import urllib.error
import urllib.request


def download_vg_image(image_id: str, output_path: str, timeout: int = 30) -> bool:
    """
    GQA images are Visual Genome images split across VG_100K and VG_100K_2.
    Download only the subset needed by the jsonl files.
    """
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return True

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    urls = [
        f"https://cs.stanford.edu/people/rak248/VG_100K/{image_id}.jpg",
        f"https://cs.stanford.edu/people/rak248/VG_100K_2/{image_id}.jpg",
    ]

    tmp_path = output_path + ".part"
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as src, open(tmp_path, "wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
            if os.path.getsize(tmp_path) > 0:
                os.replace(tmp_path, output_path)
                return True
        except (urllib.error.URLError, TimeoutError, OSError):
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            continue
    return False
