#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(readr)
  library(scales)
  library(tidyr)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2 || length(args) > 3) {
  stop("Usage: analyze_and_plot_results_v4.R PROJECT3_ROOT OUTPUT_DIR [RESULTS_MANIFEST]")
}

project_root <- normalizePath(args[[1]], mustWork = TRUE)
output_dir <- args[[2]]
manifest_path <- if (length(args) == 3) {
  args[[3]]
} else {
  file.path(project_root, "config", "results_manifest.conf")
}
manifest_path <- normalizePath(manifest_path, mustWork = TRUE)
figure_dir <- file.path(output_dir, "figures_v4")
table_dir <- file.path(output_dir, "tables_v4")
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)

read_manifest <- function(path) {
  lines <- trimws(readLines(path, warn = FALSE))
  lines <- lines[nzchar(lines) & !startsWith(lines, "#")]
  separators <- regexpr("=", lines, fixed = TRUE)
  if (length(lines) == 0 || any(separators < 2)) {
    stop("Results manifest must contain key=value entries")
  }

  keys <- trimws(substr(lines, 1, separators - 1))
  values <- trimws(substr(lines, separators + 1, nchar(lines)))
  if (any(!nzchar(keys)) || any(!nzchar(values))) {
    stop("Results manifest contains an empty key or value")
  }
  if (anyDuplicated(keys)) {
    stop(paste("Results manifest contains duplicate keys:",
               paste(unique(keys[duplicated(keys)]), collapse = ", ")))
  }
  setNames(as.list(values), keys)
}

resolve_manifest_entry <- function(value, key) {
  entries <- trimws(strsplit(value, ",", fixed = TRUE)[[1]])
  if (any(!nzchar(entries))) {
    stop(paste("Manifest entry", key, "contains an empty filename"))
  }

  matches <- unlist(lapply(entries, function(entry) {
    Sys.glob(file.path(project_root, entry))
  }), use.names = FALSE)
  matches <- unique(matches)
  if (length(matches) == 0) {
    stop(paste("Manifest entry", key, "did not match any files:", value))
  }
  normalizePath(matches, mustWork = TRUE)
}

manifest <- read_manifest(manifest_path)
required_keys <- c(
  "serial", "serial_scaling", "strong_one_node", "strong_two_node",
  "strong_256", "hybrid_one_node", "hybrid_two_node", "weak",
  "openmp_gpu", "openmp_gpu_cpu", "pokec_analysis"
)
missing_keys <- setdiff(required_keys, names(manifest))
if (length(missing_keys) > 0) {
  stop(paste("Results manifest is missing keys:", paste(missing_keys, collapse = ", ")))
}

paths <- setNames(lapply(required_keys, function(key) {
  resolve_manifest_entry(manifest[[key]], key)
}), required_keys)

if (length(paths$pokec_analysis) != 1) {
  stop("Manifest key pokec_analysis must resolve to exactly one CSV file")
}

read_many <- function(files) {
  if (length(files) == 0) return(tibble())
  bind_rows(lapply(files, function(path) {
    read_csv(path, show_col_types = FALSE) %>% mutate(source_file = basename(path))
  }))
}

validate_numerics <- function(data, label) {
  valid <- data$iterations == 100 & data$total_seconds > 0 &
    data$min_rank >= -1e-15 & abs(data$rank_sum - 1) <= 1e-6 &
    is.finite(data$total_seconds) & is.finite(data$residual)
  if (!all(valid)) stop(paste(label, "contains invalid numerical rows"))
}

validate_repetitions <- function(data, keys, expected, label) {
  audit <- data %>% count(across(all_of(keys)), name = "repetitions")
  if (!all(audit$repetitions == expected)) {
    stop(paste(label, "does not have", expected, "repetitions for every configuration"))
  }
  audit
}

summarise_runs <- function(data, keys) {
  data %>%
    group_by(across(all_of(keys))) %>%
    summarise(
      repetitions = n(),
      total_seconds_median = median(total_seconds),
      total_seconds_min = min(total_seconds),
      total_seconds_max = max(total_seconds),
      total_seconds_mean = mean(total_seconds),
      total_seconds_sd = sd(total_seconds),
      total_seconds_cv = sd(total_seconds) / mean(total_seconds),
      setup_seconds_median = median(setup_seconds),
      compute_seconds_median = median(compute_seconds),
      communication_seconds_median = median(communication_seconds),
      allgather_seconds_median = median(allgather_seconds),
      allreduce_seconds_median = median(allreduce_seconds),
      edges_per_second_median = median(edges_per_second),
      local_vertices_min = median(local_vertices_min),
      local_vertices_max = median(local_vertices_max),
      local_edges_min = median(local_edges_min),
      local_edges_max = median(local_edges_max),
      rank_sum_max_error = max(abs(rank_sum - 1)),
      residual_max = max(residual),
      .groups = "drop"
    ) %>%
    mutate(
      phase_sum = compute_seconds_median + communication_seconds_median,
      compute_phase_share = if_else(phase_sum > 0, compute_seconds_median / phase_sum, NA_real_),
      communication_phase_share = if_else(phase_sum > 0, communication_seconds_median / phase_sum, NA_real_),
      edge_imbalance = if_else(local_edges_min > 0, local_edges_max / local_edges_min, NA_real_),
      vertex_imbalance = if_else(local_vertices_min > 0, local_vertices_max / local_vertices_min, NA_real_)
    )
}

serial_raw <- read_many(paths$serial)
serial_scaling_raw <- read_many(paths$serial_scaling)
strong_one_raw <- read_many(paths$strong_one_node)
strong_two_raw <- read_many(paths$strong_two_node)
strong_256_raw <- read_many(paths$strong_256)
hybrid_one_raw <- read_many(paths$hybrid_one_node)
hybrid_two_raw <- read_many(paths$hybrid_two_node)
weak_raw <- read_many(paths$weak)
gpu_raw <- bind_rows(read_many(paths$openmp_gpu), read_many(paths$openmp_gpu_cpu))

invisible(lapply(list(serial = serial_raw, serial_scaling = serial_scaling_raw,
            strong_one = strong_one_raw, strong_two = strong_two_raw,
            hybrid_one = hybrid_one_raw, hybrid_two = hybrid_two_raw,
            weak = weak_raw, gpu = gpu_raw),
       function(x) validate_numerics(x, "curated data")))
if (nrow(strong_256_raw) > 0) validate_numerics(strong_256_raw, "strong 256")

audits <- bind_rows(
  validate_repetitions(serial_raw, c("program", "dataset"), 3, "serial") %>% mutate(experiment = "serial"),
  validate_repetitions(serial_scaling_raw, c("vertices", "edges"), 3, "serial scaling") %>% mutate(experiment = "serial scaling"),
  validate_repetitions(strong_one_raw, c("partition", "mpi_processes", "omp_threads"), 3, "strong one node") %>% mutate(experiment = "strong one node"),
  validate_repetitions(strong_two_raw, c("partition", "mpi_processes", "omp_threads"), 3, "strong two node") %>% mutate(experiment = "strong two node"),
  if (nrow(strong_256_raw) > 0) {
    validate_repetitions(strong_256_raw, c("partition", "mpi_processes", "omp_threads"), 3, "strong four node") %>%
      mutate(experiment = "strong four node")
  } else tibble(),
  validate_repetitions(hybrid_one_raw, c("partition", "mpi_processes", "omp_threads"), 3, "hybrid one node") %>% mutate(experiment = "hybrid one node"),
  validate_repetitions(hybrid_two_raw, c("partition", "mpi_processes", "omp_threads"), 3, "hybrid two node") %>% mutate(experiment = "hybrid two node"),
  validate_repetitions(weak_raw, c("vertices", "edges", "mpi_processes"), 3, "weak") %>% mutate(experiment = "weak"),
  validate_repetitions(gpu_raw, c("program", "dataset"), 3, "OpenMP GPU") %>% mutate(experiment = "OpenMP GPU")
)

serial <- summarise_runs(serial_raw, c("program", "dataset"))
serial_scaling <- summarise_runs(serial_scaling_raw, c("dataset", "vertices", "edges")) %>%
  mutate(edges_per_vertex = edges / vertices)

strong_one <- summarise_runs(strong_one_raw, c("partition", "mpi_processes", "omp_threads"))
strong_two <- summarise_runs(strong_two_raw, c("partition", "mpi_processes", "omp_threads"))
strong_256 <- if (nrow(strong_256_raw) > 0) {
  summarise_runs(strong_256_raw, c("partition", "mpi_processes", "omp_threads"))
} else tibble()
strong <- bind_rows(strong_one, strong_two, strong_256) %>%
  group_by(partition) %>%
  mutate(
    baseline_seconds = total_seconds_median[mpi_processes == 1][1],
    speedup = baseline_seconds / total_seconds_median,
    parallel_efficiency = speedup / mpi_processes
  ) %>% ungroup() %>% arrange(partition, mpi_processes)

hybrid_one <- summarise_runs(hybrid_one_raw, c("partition", "mpi_processes", "omp_threads")) %>%
  mutate(configuration = paste0(mpi_processes, "x", omp_threads), cores = mpi_processes * omp_threads)
hybrid_two <- summarise_runs(hybrid_two_raw, c("partition", "mpi_processes", "omp_threads")) %>%
  mutate(configuration = paste0(mpi_processes, "x", omp_threads), cores = mpi_processes * omp_threads)

weak <- summarise_runs(weak_raw, c("dataset", "vertices", "edges", "partition", "mpi_processes", "omp_threads")) %>%
  arrange(mpi_processes) %>%
  mutate(
    baseline_seconds = total_seconds_median[mpi_processes == 1][1],
    weak_efficiency = baseline_seconds / total_seconds_median,
    runtime_growth = total_seconds_median / baseline_seconds,
    vertices_per_rank = vertices / mpi_processes,
    edges_per_rank = edges / mpi_processes
  )

gpu <- summarise_runs(gpu_raw, c("program", "dataset"))
gpu_cpu_time <- gpu$total_seconds_median[gpu$program == "serial"][[1]]
gpu_device_time <- gpu$total_seconds_median[gpu$program == "openmp_gpu"][[1]]
gpu <- gpu %>% mutate(speedup_vs_gpu_node_serial = gpu_cpu_time / total_seconds_median)

write_csv(audits, file.path(table_dir, "configuration_audit_v4.csv"))
write_csv(serial, file.path(table_dir, "serial_v4.csv"))
write_csv(serial_scaling, file.path(table_dir, "serial_scaling_v4.csv"))
write_csv(strong, file.path(table_dir, "strong_v4.csv"))
write_csv(hybrid_one, file.path(table_dir, "hybrid_one_node_v4.csv"))
write_csv(hybrid_two, file.path(table_dir, "hybrid_two_node_v4.csv"))
write_csv(weak, file.path(table_dir, "weak_v4.csv"))
write_csv(gpu, file.path(table_dir, "openmp_gpu_v4.csv"))

theme_report <- function() {
  theme_minimal(base_size = 11, base_family = "sans") +
    theme(plot.title = element_text(face = "bold", size = 13),
          plot.subtitle = element_text(color = "#4D4D4D", size = 9.5),
          legend.position = "top", panel.grid.minor = element_blank(),
          plot.background = element_rect(fill = "white", color = NA),
          panel.background = element_rect(fill = "white", color = NA))
}
partition_colors <- c(vertices = "#0072B2", edges = "#D55E00")
save_plot <- function(name, plot, width = 7.4, height = 4.6) {
  ggsave(file.path(figure_dir, name), plot, width = width, height = height,
         dpi = 300, bg = "white")
}

vertex_series <- serial_scaling %>% filter(abs(edges_per_vertex - 16) < 1e-9) %>% arrange(vertices)
vertex_fit <- lm(log(total_seconds_median) ~ log(vertices), data = vertex_series)
vertex_exponent <- unname(coef(vertex_fit)[[2]])
vertex_series <- vertex_series %>% mutate(fitted_seconds = exp(predict(vertex_fit)))
p <- ggplot(vertex_series, aes(vertices, total_seconds_median)) +
  geom_ribbon(aes(ymin = total_seconds_min, ymax = total_seconds_max), fill = "#0072B2", alpha = 0.14) +
  geom_line(color = "#0072B2", linewidth = 0.85) + geom_point(color = "#0072B2", size = 2.3) +
  geom_line(aes(y = fitted_seconds), color = "#444444", linetype = "dashed", linewidth = 0.75) +
  scale_x_log10(labels = label_number(big.mark = ",")) + scale_y_log10(labels = label_number()) +
  labs(title = "Serial runtime versus graph vertices",
       subtitle = sprintf("16 edges per vertex; dashed fit has exponent %.2f; range shows three repetitions", vertex_exponent),
       x = "Vertices (log scale)", y = "Runtime (s, log scale)") + theme_report()
save_plot("serial_vertices_v4.png", p)

edge_series <- serial_scaling %>% filter(vertices == 1000000) %>% arrange(edges)
edge_fit <- lm(log(total_seconds_median) ~ log(edges), data = edge_series)
edge_exponent <- unname(coef(edge_fit)[[2]])
edge_series <- edge_series %>% mutate(fitted_seconds = exp(predict(edge_fit)))
p <- ggplot(edge_series, aes(edges, total_seconds_median)) +
  geom_ribbon(aes(ymin = total_seconds_min, ymax = total_seconds_max), fill = "#009E73", alpha = 0.14) +
  geom_line(color = "#009E73", linewidth = 0.85) + geom_point(color = "#009E73", size = 2.3) +
  geom_line(aes(y = fitted_seconds), color = "#444444", linetype = "dashed", linewidth = 0.75) +
  scale_x_log10(labels = label_number(big.mark = ",")) + scale_y_log10(labels = label_number()) +
  labs(title = "Serial runtime versus graph edges",
       subtitle = sprintf("One million vertices; dashed fit has exponent %.2f; range shows three repetitions", edge_exponent),
       x = "Edges (log scale)", y = "Runtime (s, log scale)") + theme_report()
save_plot("serial_edges_v4.png", p)

strong_breaks <- sort(unique(strong$mpi_processes))
p <- ggplot(strong, aes(mpi_processes, total_seconds_median, color = partition, fill = partition)) +
  geom_ribbon(aes(ymin = total_seconds_min, ymax = total_seconds_max), alpha = 0.12, linewidth = 0) +
  geom_line(linewidth = 0.85) + geom_point(size = 2.3) +
  scale_x_continuous(trans = "log2", breaks = strong_breaks) + scale_y_log10() +
  scale_color_manual(values = partition_colors) + scale_fill_manual(values = partition_colors) +
  labs(title = "Strong-scaling runtime", subtitle = "Pokec, 100 iterations; results through 256 ranks",
       x = "MPI ranks / CPU cores (log scale)", y = "Runtime (s, log scale)",
       color = "Partition", fill = "Partition") + theme_report()
save_plot("strong_runtime_v4.png", p)

ideal <- tibble(mpi_processes = strong_breaks, speedup = strong_breaks)
p <- ggplot(strong, aes(mpi_processes, speedup, color = partition)) +
  geom_line(linewidth = 0.85) + geom_point(size = 2.3) +
  geom_line(data = ideal, aes(mpi_processes, speedup), inherit.aes = FALSE,
            color = "#666666", linetype = "dashed", linewidth = 0.7) +
  scale_x_continuous(trans = "log2", breaks = strong_breaks) + scale_y_log10() +
  scale_color_manual(values = partition_colors) +
  labs(title = "Strong-scaling speedup", subtitle = "Dashed line shows ideal speedup; results through 256 ranks",
       x = "MPI ranks / CPU cores (log scale)", y = "Speedup (log scale)", color = "Partition") + theme_report()
save_plot("strong_speedup_v4.png", p)

p <- ggplot(strong, aes(mpi_processes, parallel_efficiency, color = partition)) +
  geom_line(linewidth = 0.85) + geom_point(size = 2.3) +
  geom_hline(yintercept = 1, color = "#666666", linetype = "dashed", linewidth = 0.7) +
  scale_x_continuous(trans = "log2", breaks = strong_breaks) +
  scale_y_continuous(labels = label_percent()) + scale_color_manual(values = partition_colors) +
  labs(title = "Strong-scaling efficiency", subtitle = "Efficiency uses the matching one-rank MPI baseline",
       x = "MPI ranks / CPU cores (log scale)", y = "Parallel efficiency", color = "Partition") + theme_report()
save_plot("strong_efficiency_v4.png", p)

p <- ggplot(strong, aes(mpi_processes, communication_phase_share, color = partition)) +
  geom_line(linewidth = 0.85) + geom_point(size = 2.3) +
  scale_x_continuous(trans = "log2", breaks = strong_breaks) +
  scale_y_continuous(labels = label_percent()) + scale_color_manual(values = partition_colors) +
  labs(title = "Normalized communication phase weight",
       subtitle = "Phase maxima are diagnostic and are not additive wall-time components",
       x = "MPI ranks / CPU cores (log scale)", y = "Communication / phase sum", color = "Partition") + theme_report()
save_plot("strong_communication_v4.png", p)

hybrid_order <- c("72x1", "36x2", "18x4", "9x8", "6x12", "4x18", "2x36", "1x72")
hybrid_plot <- hybrid_one %>% mutate(configuration = factor(configuration, levels = hybrid_order))
p <- ggplot(hybrid_plot, aes(configuration, total_seconds_median, color = partition, group = partition)) +
  geom_line(linewidth = 0.85) + geom_point(size = 2.4) +
  scale_y_log10() + scale_color_manual(values = partition_colors) +
  labs(title = "One-node MPI/OpenMP configuration sweep",
       subtitle = "Every configuration uses 72 CPU cores",
       x = "MPI ranks x OpenMP threads", y = "Median runtime (s, log scale)", color = "Partition") +
  theme_report() + theme(axis.text.x = element_text(angle = 35, hjust = 1))
save_plot("hybrid_one_node_v4.png", p)

phase_plot <- hybrid_one %>%
  select(partition, configuration, compute_phase_share, communication_phase_share) %>%
  pivot_longer(c(compute_phase_share, communication_phase_share), names_to = "phase", values_to = "share") %>%
  mutate(configuration = factor(configuration, levels = hybrid_order),
         phase = recode(phase, compute_phase_share = "Computation", communication_phase_share = "Communication"))
p <- ggplot(phase_plot, aes(configuration, share, fill = phase)) +
  geom_col() + facet_wrap(~partition, ncol = 1) + scale_y_continuous(labels = label_percent()) +
  scale_fill_manual(values = c(Computation = "#009E73", Communication = "#CC79A7")) +
  labs(title = "One-node normalized phase weights", subtitle = "Independent phase maxima normalized by their sum",
       x = "MPI ranks x OpenMP threads", y = "Normalized phase weight", fill = "Phase") +
  theme_report() + theme(axis.text.x = element_text(angle = 35, hjust = 1))
save_plot("hybrid_phase_v4.png", p, 7.4, 6.1)

two_order <- c("144x1", "72x2", "36x4", "18x8", "8x18", "4x36", "2x72")
two_plot <- hybrid_two %>% mutate(configuration = factor(configuration, levels = two_order))
p <- ggplot(two_plot, aes(configuration, total_seconds_median, color = partition, group = partition)) +
  geom_line(linewidth = 0.85) + geom_point(size = 2.4) + scale_y_log10() +
  scale_color_manual(values = partition_colors) +
  labs(title = "Two-node MPI/OpenMP configuration sweep",
       subtitle = "Every configuration uses 144 CPU cores across two nodes",
       x = "MPI ranks x OpenMP threads", y = "Median runtime (s, log scale)", color = "Partition") +
  theme_report() + theme(axis.text.x = element_text(angle = 35, hjust = 1))
save_plot("hybrid_two_node_v4.png", p)

weak_breaks <- weak$mpi_processes
p <- ggplot(weak, aes(mpi_processes, total_seconds_median)) +
  geom_ribbon(aes(ymin = total_seconds_min, ymax = total_seconds_max), fill = "#0072B2", alpha = 0.13) +
  geom_line(color = "#0072B2", linewidth = 0.85) + geom_point(color = "#0072B2", size = 2.3) +
  geom_hline(yintercept = weak$baseline_seconds[[1]], color = "#666666", linetype = "dashed") +
  scale_x_continuous(trans = "log2", breaks = weak_breaks) + scale_y_log10() +
  labs(title = "Weak-scaling runtime", subtitle = "100,000 vertices and 1.6 million edges per rank; results through 256 ranks",
       x = "MPI ranks / CPU cores (log scale)", y = "Runtime (s, log scale)") + theme_report()
save_plot("weak_runtime_v4.png", p)

p <- ggplot(weak, aes(mpi_processes, weak_efficiency)) +
  geom_line(color = "#0072B2", linewidth = 0.85) + geom_point(color = "#0072B2", size = 2.3) +
  geom_hline(yintercept = 1, color = "#666666", linetype = "dashed") +
  scale_x_continuous(trans = "log2", breaks = weak_breaks) +
  scale_y_continuous(labels = label_percent()) +
  labs(title = "Weak-scaling efficiency", subtitle = "T(1)/T(P); results through 256 ranks",
       x = "MPI ranks / CPU cores (log scale)", y = "Weak-scaling efficiency") + theme_report()
save_plot("weak_efficiency_v4.png", p)

gpu_plot <- gpu %>% mutate(program_label = recode(program, serial = "Serial CPU", openmp_gpu = "OpenMP GPU"))
p <- ggplot(gpu_plot, aes(program_label, total_seconds_median, color = program_label)) +
  geom_linerange(aes(ymin = total_seconds_min, ymax = total_seconds_max), linewidth = 1.1) +
  geom_point(size = 3.2) + scale_y_log10() +
  scale_color_manual(values = c("Serial CPU" = "#666666", "OpenMP GPU" = "#009E73"), guide = "none") +
  labs(title = "OpenMP GPU offload on NVIDIA A40", subtitle = "Pokec, 100 iterations; CPU reference measured on the same node",
       x = NULL, y = "Runtime (s, log scale)") + theme_report()
save_plot("openmp_gpu_v4.png", p, 6.2, 4.4)

pokec <- read_csv(paths$pokec_analysis[[1]], show_col_types = FALSE)
value <- function(metric) as.numeric(pokec$value[pokec$metric == metric][[1]])
concentration <- tibble(
  group = factor(c("Top 1%", "Top 5%", "Top 10%"), levels = c("Top 1%", "Top 5%", "Top 10%")),
  share = c(value("top_1_percent_pagerank_share"), value("top_5_percent_pagerank_share"),
            value("top_10_percent_pagerank_share"))
)
p <- ggplot(concentration, aes(group, share)) +
  geom_col(fill = "#0072B2", width = 0.62) +
  geom_text(aes(label = percent(share, accuracy = 0.1)), vjust = -0.4, size = 3.5) +
  scale_y_continuous(labels = label_percent(), limits = c(0, max(concentration$share) * 1.18)) +
  labs(title = "Concentration of PageRank in Pokec", x = NULL, y = "Share of total PageRank") + theme_report()
save_plot("pokec_concentration_v4.png", p, 6.3, 4.3)

high_variability <- bind_rows(
  serial_scaling %>% mutate(experiment = "serial scaling", configuration = dataset),
  strong %>% mutate(experiment = "strong", configuration = paste(partition, mpi_processes, sep = "-")),
  hybrid_one %>% mutate(experiment = "hybrid one node", configuration = paste(partition, configuration, sep = "-")),
  hybrid_two %>% mutate(experiment = "hybrid two node", configuration = paste(partition, configuration, sep = "-")),
  weak %>% mutate(experiment = "weak", configuration = paste0(mpi_processes, " ranks")),
  gpu %>% mutate(experiment = "GPU", configuration = program)
) %>%
  select(experiment, configuration, repetitions, total_seconds_median,
         total_seconds_min, total_seconds_max, total_seconds_cv) %>%
  filter(total_seconds_cv > 0.15) %>% arrange(desc(total_seconds_cv))
write_csv(high_variability, file.path(table_dir, "high_variability_v4.csv"))

cat("Validated curated primary data.\n")
cat("Results manifest:", manifest_path, "\n")
cat("Strong scaling maximum rank:", max(strong$mpi_processes), "\n")
cat("Weak scaling maximum rank:", max(weak$mpi_processes), "\n")
cat("OpenMP GPU speedup:", round(gpu_cpu_time / gpu_device_time, 3), "\n")
cat("Figures:", figure_dir, "\n")
cat("Tables:", table_dir, "\n")
