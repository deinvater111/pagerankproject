#!/usr/bin/env python3
"""Create a compact structural summary from a converged Pokec PageRank vector."""

import argparse
import csv
import heapq
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--top", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    scores = []
    top = []
    count = 0
    dangling = 0
    edge_count = 0
    rank_sum = 0.0
    minimum_rank = math.inf
    maximum_rank = 0.0
    sum_indegree = 0.0
    sum_indegree_squared = 0.0
    sum_rank_squared = 0.0
    sum_cross = 0.0

    with open(args.scores, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            original_id = int(row["original_id"])
            score = float(row["pagerank"])
            indegree = int(row["indegree"])
            outdegree = int(row["outdegree"])
            scores.append(score)
            item = (score, original_id, indegree, outdegree)
            if len(top) < args.top_k:
                heapq.heappush(top, item)
            elif item > top[0]:
                heapq.heapreplace(top, item)

            count += 1
            dangling += int(outdegree == 0)
            edge_count += indegree
            rank_sum += score
            minimum_rank = min(minimum_rank, score)
            maximum_rank = max(maximum_rank, score)
            sum_indegree += indegree
            sum_indegree_squared += indegree * indegree
            sum_rank_squared += score * score
            sum_cross += indegree * score

    if count == 0:
        raise SystemExit("Score file is empty")

    scores.sort(reverse=True)
    concentration = {}
    for percent in (1, 5, 10):
        take = max(1, math.ceil(count * percent / 100.0))
        concentration[percent] = sum(scores[:take])

    covariance_numerator = sum_cross - (sum_indegree * rank_sum / count)
    indegree_variance = sum_indegree_squared - (sum_indegree * sum_indegree / count)
    rank_variance = sum_rank_squared - (rank_sum * rank_sum / count)
    denominator = math.sqrt(max(0.0, indegree_variance) * max(0.0, rank_variance))
    pearson = covariance_numerator / denominator if denominator else float("nan")

    metrics = [
        ("vertices", count),
        ("edges", edge_count),
        ("average_outdegree", edge_count / count),
        ("directed_density_no_self_edges", edge_count / (count * (count - 1))),
        ("dangling_vertices", dangling),
        ("dangling_fraction", dangling / count),
        ("pagerank_sum", rank_sum),
        ("minimum_pagerank", minimum_rank),
        ("maximum_pagerank", maximum_rank),
        ("top_1_percent_pagerank_share", concentration[1]),
        ("top_5_percent_pagerank_share", concentration[5]),
        ("top_10_percent_pagerank_share", concentration[10]),
        ("pearson_pagerank_indegree", pearson),
    ]

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(metrics)

    top_path = Path(args.top)
    top_path.parent.mkdir(parents=True, exist_ok=True)
    with top_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "original_id", "pagerank", "indegree", "outdegree"])
        for position, item in enumerate(sorted(top, reverse=True), start=1):
            score, original_id, indegree, outdegree = item
            writer.writerow([position, original_id, f"{score:.17g}", indegree, outdegree])

    print(f"Wrote {summary_path}")
    print(f"Wrote {top_path}")


if __name__ == "__main__":
    main()
