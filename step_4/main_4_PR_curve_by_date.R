library(dplyr)
library(ggplot2)
library(patchwork)

ROOT     <- "/Users/takashi/LocalAnalysis/WeedMap"
PROB_DIR <- file.path(ROOT, "Results_TabICL", "NNDM_prob")
OUTDIR   <- file.path(ROOT, "Results_TabICL", "figures")
TARGET   <- "log1p_Chenopodium_Count"
T_VAL    <- 10

ALG_FILES <- c(
  "OK"          = file.path(PROB_DIR, sprintf("NNDM_OK_%s_taucurve_T%d.csv",            TARGET, T_VAL)),
  "TabICL"      = file.path(PROB_DIR, sprintf("NNDM_Vanilla_TabPFN_%s_taucurve_T%d.csv", TARGET, T_VAL)),
  "TabICL-KpR"  = file.path(PROB_DIR, sprintf("NNDM_KpR_%s_taucurve_T%d.csv",           TARGET, T_VAL)),
  "TabICL-RK"   = file.path(PROB_DIR, sprintf("NNDM_RK_%s_taucurve_T%d.csv",            TARGET, T_VAL))
)

ALG_COLORS <- c(
  "OK"         = "#4477AA",
  "TabICL"     = "#009999",
  "TabICL-KpR" = "#cc6699",
  "TabICL-RK"  = "#e6b800"
)
ALG_LEVELS <- c("OK", "TabICL", "TabICL-KpR", "TabICL-RK")

TAU_LABEL_VALS <- c(0.1, 0.3, 0.5, 0.7, 0.9)

DATE_LABELS <- c(
  "20250424" = "24 Apr", "20250430" = "30 Apr",
  "20250506" = "06 May", "20250513" = "13 May",
  "20250520" = "20 May", "20250526" = "26 May",
  "20250602" = "02 Jun"
)

# ---------------------------
# Load and combine
# ---------------------------
dat <- bind_rows(lapply(names(ALG_FILES), function(alg) {
  fp <- ALG_FILES[[alg]]
  if (!file.exists(fp)) { warning("Missing: ", fp); return(NULL) }
  read.csv(fp, stringsAsFactors = FALSE) %>%
    mutate(algorithm = alg)
})) %>%
  filter(prevalence > 0) %>%
  filter(date %in% names(DATE_LABELS)) %>%
  filter(tau > 0.05 & tau < 0.95) %>%
  filter(!(recall == 0 & precision == 0)) %>%
  mutate(
    algorithm  = factor(algorithm, levels = ALG_LEVELS),
    date_label = factor(DATE_LABELS[date], levels = DATE_LABELS)
  )

# τ annotation points
tau_pts <- dat %>%
  filter(abs(tau - round(tau / 0.1) * 0.1) < 0.005,
         tau %in% TAU_LABEL_VALS)

# ---------------------------
# One panel per date
# ---------------------------
make_pr_panel <- function(df_date, df_tau, date_lbl, show_x, show_y) {
  p <- ggplot(df_date, aes(x = recall, y = precision,
                            colour = algorithm, group = algorithm)) +
    geom_path(linewidth = 0.9) +
    scale_colour_manual(values = ALG_COLORS, name = "Method") +
    scale_x_continuous(limits = c(0, 1), breaks = c(0, 0.5, 1),
                       labels = c("0", "0.5", "1")) +
    scale_y_continuous(limits = c(0, 1), breaks = c(0, 0.5, 1),
                       labels = c("0", "0.5", "1")) +
    labs(
      title = date_lbl,
      x     = if (show_x) "Recall" else NULL,
      y     = if (show_y) "Precision" else NULL
    ) +
    theme_bw(base_size = 16) +
    theme(
      plot.title       = element_text(size = 16, face = "bold", hjust = 0.5),
      legend.position  = "none",
      panel.grid.minor = element_blank(),
      axis.text        = element_text(size = 14),
      axis.title       = element_text(size = 15),
      plot.margin      = unit(c(1, 2, 1, 2), "mm")
    )
  p
}

dates_ordered <- names(DATE_LABELS)
panels <- list()
for (i in seq_along(dates_ordered)) {
  d      <- dates_ordered[i]
  lbl    <- DATE_LABELS[d]
  row_i  <- ceiling(i / 3)
  col_i  <- ((i - 1) %% 3) + 1
  show_x <- row_i >= 2
  show_y <- col_i == 1

  df_d   <- dat      %>% filter(date == d)
  df_tau <- tau_pts  %>% filter(date == d)

  if (nrow(df_d) == 0) next
  panels[[d]] <- make_pr_panel(df_d, df_tau, lbl, show_x, show_y)
}

# 7 panels in 3×3 grid — two blank cells at positions 8 and 9
combined <- wrap_plots(c(panels, list(plot_spacer(), plot_spacer())), ncol = 3) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

ggsave(file.path(OUTDIR, "PR_curve_by_date.png"),
       plot = combined, width = 11, height = 11, dpi = 150)
ggsave(file.path(OUTDIR, "PR_curve_by_date.tiff"),
       plot = combined, width = 11, height = 11, dpi = 300, device = "tiff")
cat("Saved PR_curve_by_date\n")
