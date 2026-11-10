#!/usr/bin/env python3
"""Compare two native-double PageRank vectors in bounded memory."""

import argparse
import array
import csv
from pathlib import Path


def read_chunk(handle, values_per_chunk):
    data = handle.read(values_per_chunk * 8)
    if not data:
        return None
    if len(data) % 8:
        raise SystemExit("Score file size is not a multiple of eight bytes")
    values = array.array("d")
    values.frombytes(data)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--l1-tolerance", type=float, default=1e-8)
    parser.add_argument("--max-tolerance", type=float, default=1e-10)
    args = parser.parse_args()

    count = 0
    l1_difference = 0.0
    max_difference = 0.0
    reference_sum = 0.0
    candidate_sum = 0.0

    with open(args.reference, "rb") as reference_handle, open(args.candidate, "rb") as candidate_handle:
        while True:
            reference = read_chunk(reference_handle, 65536)
            candidate = read_chunk(candidate_handle, 65536)
            if reference is None and candidate is None:
                break
            if reference is None or candidate is None or len(reference) != len(candidate):
                raise SystemExit("Score files contain different numbers of values")
            for reference_score, candidate_score in zip(reference, candidate):
                difference = abs(reference_score - candidate_score)
                l1_difference += difference
                max_difference = max(max_difference, difference)
                reference_sum += reference_score
                candidate_sum += candidate_score
                count += 1

    passed = l1_difference <= args.l1_tolerance and max_difference <= args.max_tolerance
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "vertices", "l1_difference", "max_abs_difference", "reference_sum",
            "candidate_sum", "l1_tolerance", "max_tolerance", "passed",
        ])
        writer.writerow([
            count, f"{l1_difference:.17g}", f"{max_difference:.17g}",
            f"{reference_sum:.17g}", f"{candidate_sum:.17g}",
            args.l1_tolerance, args.max_tolerance, int(passed),
        ])

    print(f"Compared {count} binary scores")
    print(f"L1 difference: {l1_difference:.17g}")
    print(f"Maximum absolute difference: {max_difference:.17g}")
    print(f"Validation passed: {passed}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
