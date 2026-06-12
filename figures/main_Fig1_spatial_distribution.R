# ============================================================
# Figure 1 — Spatial distribution of raw Chenopodium observations
# Dates with prevalence >= 20% are auto-selected
# Output: Results_TabICL/figures/Figure1_spatial_distribution_Chenopodium.{png,tiff}
# ============================================================

library(tidyverse)
library(sf)

ROOT     <- "/Users/takashi/LocalAnalysis/WeedMap"
UAV_ROOT <- file.path(ROOT, "data")
out_dir  <- file.path(ROOT, "Results_TabICL", "figures")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

DATES <- c("20250414","20250424","20250430","20250506","20250513","20250520","20250526","20250602")

# ------------------------------------------------------------
# 1. Load per-date result.csv files (already contain Chenopodium_Count)
# ------------------------------------------------------------
result_list <- lapply(DATES, function(d) {
  uav_tag <- paste0(substr(d, 3, 8), "F3mRX")
  csv     <- file.path(UAV_ROOT, uav_tag, "result.csv")
  df      <- read.csv(csv, stringsAsFactors = FALSE)
  df$Date <- format(as.Date(d, "%Y%m%d"), "%Y-%m-%d")
  df
})
result <- do.call(rbind, result_list)

# ------------------------------------------------------------
# 2. Per-date stats — select dates with prevalence >= 20%
# ------------------------------------------------------------
date_stats <- result %>%
  group_by(Date) %>%
  summarise(
    n          = n(),
    prevalence = mean(Chenopodium_Count > 0),
    mean_count = mean(Chenopodium_Count),
    max_count  = max(Chenopodium_Count),
    .groups    = "drop"
  ) %>%
  arrange(Date)

cat("\n=== Per-date statistics ===\n")
print(as.data.frame(date_stats))

selected_dates <- date_stats %>%
  filter(prevalence >= 0.0) %>%
  pull(Date)

cat("\nSelected dates (all dates):\n")
cat(paste(selected_dates, collapse = ", "), "\n\n")

result <- result %>% filter(Date %in% selected_dates)

# ------------------------------------------------------------
# 3. Transform WGS84 -> EPSG:25832
# ------------------------------------------------------------
pts     <- st_as_sf(result, coords = c("Longitude", "Latitude"), crs = 4326)
pts_utm <- st_transform(pts, 25832)
coords  <- st_coordinates(pts_utm)
result$x_25832 <- coords[, "X"]
result$y_25832 <- coords[, "Y"]

# Anonymise: shift so min(x) = 0, min(y) = 0
result$x_25832 <- result$x_25832 - min(result$x_25832)
result$y_25832 <- result$y_25832 - min(result$y_25832)

# ------------------------------------------------------------
# 4. Facet label: "30 Apr", "06 May", etc.
# ------------------------------------------------------------
result <- result %>%
  mutate(date_label = format(as.Date(Date), "%d %b"))

# Order facets chronologically
date_order <- result %>%
  distinct(Date, date_label) %>%
  arrange(Date) %>%
  pull(date_label)
result$date_label <- factor(result$date_label, levels = date_order)

# ------------------------------------------------------------
# 5. Plot
# ------------------------------------------------------------
zeros    <- result %>% filter(Chenopodium_Count == 0)
non_zero <- result %>% filter(Chenopodium_Count > 0)

# Bicolour scale with sharp boundary at T = 10 plants/m²
# Below threshold: dark purple -> light purple
# Above threshold: orange -> yellow  (sharp step at 10 -> 10.001)
T_thresh      <- 10
lo_val        <- log1p(0.5)
thr_val       <- log1p(T_thresh)
max_val       <- log1p(max(non_zero$Chenopodium_Count, na.rm = TRUE))
colour_values <- scales::rescale(
  c(lo_val, thr_val, thr_val + 1e-3, max_val),
  from = c(lo_val, max_val)
)
colour_stops <- c("#2D0057", "#C67BCA", "#E8641A", "#FDE725")
# dark purple -> light purple | orange -> yellow

p <- ggplot() +
  geom_point(
    data   = zeros,
    aes(x = x_25832, y = y_25832),
    colour = "grey70", size = 0.8, alpha = 0.5
  ) +
  geom_point(
    data = non_zero,
    aes(x = x_25832, y = y_25832, colour = log1p(Chenopodium_Count)),
    size = 1.5
  ) +
  scale_colour_gradientn(
    colours = colour_stops,
    values  = colour_values,
    limits  = c(lo_val, max_val),
    name    = expression(paste(italic("Chenopodium"), " density (plants m"^{-2}, ")")),
    breaks  = log1p(c(1, 5, 10, 20, 50)),
    labels  = c("1", "5", "10", "20", "50"),
    oob     = scales::squish
  ) +
  facet_wrap(~ date_label, ncol = 3) +
  coord_equal() +
  labs(
    x = "Relative Easting (m)",
    y = "Relative Northing (m)"
  ) +
  theme_bw(base_size = 11) +
  theme(
    panel.grid   = element_blank(),
    strip.text   = element_text(face = "bold", size = 11),
    legend.position = "right"
  )

# ------------------------------------------------------------
# 6. Save
# ------------------------------------------------------------
n_panels <- length(selected_dates)
fig_h <- ceiling(n_panels / 3) * 3.5 + 0.5  # ~3.5 in per row

ggsave(
  file.path(out_dir, "Figure1_spatial_distribution_Chenopodium.png"),
  plot = p, width = 10, height = fig_h, dpi = 300
)
ggsave(
  file.path(out_dir, "Figure1_spatial_distribution_Chenopodium.tiff"),
  plot = p, width = 10, height = fig_h, dpi = 300, compression = "lzw"
)

cat("Saved PNG :", file.path(out_dir, "Figure1_spatial_distribution_Chenopodium.png"), "\n")
cat("Saved TIFF:", file.path(out_dir, "Figure1_spatial_distribution_Chenopodium.tiff"), "\n")
