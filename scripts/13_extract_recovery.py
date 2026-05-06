import argparse
import os
import re


TITLE_RE = re.compile(r"^===== (?P<title>.+?) \(n=(?P<n>\d+)\),")
K_RE = re.compile(r"^K=(?P<k>\d+)$")
RECOVERY_RE = re.compile(r"^(?P<method>[A-Za-z0-9_]+)\s*:\s*(?P<value>[-+0-9.]+)%$")


def extract_recoveries(path: str, block_title: str = "All samples"):
    rows = []
    in_target_block = False
    in_recovery = False
    current_k = None
    sample_count = None

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            title_match = TITLE_RE.match(line)
            if title_match:
                in_target_block = title_match.group("title") == block_title
                in_recovery = False
                current_k = None
                sample_count = int(title_match.group("n")) if in_target_block else None
                continue

            if not in_target_block:
                continue

            k_match = K_RE.match(line)
            if k_match:
                current_k = int(k_match.group("k"))
                in_recovery = False
                continue

            if line == "Recovery ratios:":
                in_recovery = True
                continue

            if in_recovery and current_k is not None:
                recovery_match = RECOVERY_RE.match(line)
                if recovery_match:
                    rows.append(
                        {
                            "log": path,
                            "samples": sample_count,
                            "k": current_k,
                            "method": recovery_match.group("method"),
                            "recovery": float(recovery_match.group("value")),
                        }
                    )
                elif line == "" or line.startswith("====="):
                    in_recovery = False

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--block", default="All samples")
    args = parser.parse_args()

    print("log,samples,k,method,recovery_percent")
    for path in args.logs:
        for row in extract_recoveries(path, block_title=args.block):
            log_name = os.path.basename(row["log"])
            print(
                f"{log_name},{row['samples']},{row['k']},"
                f"{row['method']},{row['recovery']:.2f}"
            )


if __name__ == "__main__":
    main()
