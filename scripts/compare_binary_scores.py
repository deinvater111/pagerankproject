#!/usr/bin/env python3
"""Compare two native-double PageRank vectors in bounded memory."""

import argparse
import array
import csv
import math
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
    parser.add_argument("--expected-vertices", type=int, required=True)
    parser.add_argument("--l1-tolerance", type=float, default=1e-8)
    parser.add_argument("--max-tolerance", type=float, default=1e-10)
    parser.add_argument("--sum-tolerance", type=float, default=1e-8)
    parser.add_argument("--nonnegative-tolerance", type=float, default=1e-15)
    args = parser.parse_args()

    if args.expected_vertices <= 0:
        parser.error("--expected-vertices must be positive")
    for name in (
        "l1_tolerance", "max_tolerance", "sum_tolerance", "nonnegative_tolerance"
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")

    expected_bytes = args.expected_vertices * 8
    for label, filename in (("reference", args.reference), ("candidate", args.candidate)):
        actual_bytes = Path(filename).stat().st_size
        if actual_bytes != expected_bytes:
            raise SystemExit(
                f"{label.capitalize()} score file contains {actual_bytes // 8} complete "
                f"values ({actual_bytes} bytes); expected exactly "
                f"{args.expected_vertices} values ({expected_bytes} bytes)"
            )

    count = 0
    l1_difference = 0.0
    max_difference = 0.0
    reference_sum = 0.0
    candidate_sum = 0.0
    reference_min = math.inf
    candidate_min = math.inf
    reference_finite = True
    candidate_finite = True

    with open(args.reference, "rb") as reference_handle, open(args.candidate, "rb") as candidate_handle:
        while True:
            reference = read_chunk(reference_handle, 65536)
            candidate = read_chunk(candidate_handle, 65536)
            if reference is None and candidate is None:
                break
            if reference is None or candidate is None or len(reference) != len(candidate):
                raise SystemExit("Score files contain different numbers of values")
            for reference_score, candidate_score in zip(reference, candidate):
                reference_score_finite = math.isfinite(reference_score)
                candidate_score_finite = math.isfinite(candidate_score)
                reference_finite = reference_finite and reference_score_finite
                candidate_finite = candidate_finite and candidate_score_finite
                if reference_score_finite:
                    reference_min = min(reference_min, reference_score)
                    reference_sum += reference_score
                if candidate_score_finite:
                    candidate_min = min(candidate_min, candidate_score)
                    candidate_sum += candidate_score
                difference = (
                    abs(reference_score - candidate_score)
                    if reference_score_finite and candidate_score_finite
                    else math.inf
                )
                l1_difference += difference
                max_difference = max(max_difference, difference)
                count += 1

    count_matches = count == args.expected_vertices
    reference_nonnegative = reference_min >= -args.nonnegative_tolerance
    candidate_nonnegative = candidate_min >= -args.nonnegative_tolerance
    reference_normalized = (
        reference_finite and abs(reference_sum - 1.0) <= args.sum_tolerance
    )
    candidate_normalized = (
        candidate_finite and abs(candidate_sum - 1.0) <= args.sum_tolerance
    )
    passed = (
        count_matches
        and reference_finite
        and candidate_finite
        and reference_nonnegative
        and candidate_nonnegative
        and reference_normalized
        and candidate_normalized
        and l1_difference <= args.l1_tolerance
        and max_difference <= args.max_tolerance
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "vertices", "expected_vertices", "l1_difference", "max_abs_difference",
            "reference_sum", "candidate_sum", "reference_min", "candidate_min",
            "reference_finite", "candidate_finite", "reference_normalized",
            "candidate_normalized", "l1_tolerance", "max_tolerance",
            "sum_tolerance", "nonnegative_tolerance", "passed",
        ])
        writer.writerow([
            count, args.expected_vertices, f"{l1_difference:.17g}",
            f"{max_difference:.17g}",
            f"{reference_sum:.17g}", f"{candidate_sum:.17g}",
            f"{reference_min:.17g}", f"{candidate_min:.17g}",
            int(reference_finite), int(candidate_finite),
            int(reference_normalized), int(candidate_normalized),
            args.l1_tolerance, args.max_tolerance, args.sum_tolerance,
            args.nonnegative_tolerance, int(passed),
        ])

    print(f"Compared {count} binary scores")
    print(f"L1 difference: {l1_difference:.17g}")
    print(f"Maximum absolute difference: {max_difference:.17g}")
    print(f"Reference sum/minimum: {reference_sum:.17g} / {reference_min:.17g}")
    print(f"Candidate sum/minimum: {candidate_sum:.17g} / {candidate_min:.17g}")
    print(f"Validation passed: {passed}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
