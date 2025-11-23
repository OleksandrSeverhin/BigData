# ---
# Title: "Practice 5: Full Model Refinement (Full Dataset)"
# Author: Oleksandr Severhin
# Date: November, 2025
# ---

# --- 0. SETUP ---
if (!require("pacman")) install.packages("pacman")

pacman::p_load(
  arrow,
  dplyr,
  ggplot2,
  broom,
  scales,
  performance,
  patchwork
)

Sys.setlocale("LC_TIME", "English")

if (!dir.exists("data")) dir.create("data")
if (!dir.exists("plots")) dir.create("plots")

set.seed(123)

# --- 1. LOAD AND PREPARE DATA ---
tryCatch({
  taxi_data <- read_parquet("data/cleaned_yellow_tripdata_2025-01.parquet")
}, error = function(e) {
  stop("Error: Could not read 'data/cleaned_yellow_tripdata_2025-01.parquet'. 
       Please make sure the file exists in the 'data' directory.")
})

taxi_model_data <- taxi_data %>%
  mutate(
    log_total = log(total_amount),
    log_distance = log(trip_distance),
    log_fare = log(fare_amount),
    # Add 1 to avoid log(0) for tips and tolls
    log_tip = log(tip_amount + 1),
    log_tolls = log(tolls_amount + 1),
    # Convert categorical variables to factors
    payment_type = as.factor(payment_type),
    day_of_week = as.factor(day_of_week),
    VendorID = as.factor(VendorID)
  )

# --- 2. EDA ---
eda_plot_sample <- taxi_data_clean %>% sample_n(min(nrow(.), 100000))

fare_dist_plot <- ggplot(eda_plot_sample, aes(x = fare_amount)) +
  geom_histogram(bins = 100, fill = "#FFC400", color = "black", alpha = 0.8) +
  scale_x_continuous(labels = dollar_format()) +
  labs(
    title = "Distribution of fare_amount (Sampled for Plotting)",
    subtitle = "Data is continuous, positive, and extremely right-skewed.",
    x = "Fare Amount ($)",
    y = "Frequency"
  ) +
  theme_minimal()

print(fare_dist_plot)
ggsave("plots/01_fare_amount_distribution.png", fare_dist_plot, width = 8, height = 5)

# --- 4. MODEL 1: OLS ---
ols_model <- glm(fare_amount ~ trip_distance + passenger_count,
                 family = gaussian(link = "identity"),
                 data = taxi_data_clean)

# --- 5. MODEL 2: GAMMA GLM ---
glm_gamma_model <- glm(fare_amount ~ trip_distance + passenger_count,
                         family = Gamma(link = "log"),
                         data = taxi_data_clean)

# --- 6. MODEL COMPARISON ---
# A. AIC and Deviance
model_comp_stats <- data.frame(
  Model = c("OLS (Gaussian)", "GLM (Gamma)"),
  AIC = c(AIC(ols_model), AIC(glm_gamma_model)),
  Deviance = c(deviance(ols_model), deviance(glm_gamma_model)),
  Pseudo_R2 = c(r2(ols_model)$R2, r2(glm_gamma_model)$R2_McFadden)
)

# B. RMSE
preds_ols <- predict(ols_model, newdata = taxi_data_clean, type = "response")
preds_glm <- predict(glm_gamma_model, newdata = taxi_data_clean, type = "response")

rmse_ols <- sqrt(mean((taxi_data_clean$fare_amount - preds_ols)^2))
rmse_glm <- sqrt(mean((taxi_data_clean$fare_amount - preds_glm)^2))

rmse_comp <- data.frame(
  Model = c("OLS (Gaussian)", "GLM (Gamma)"),
  RMSE = c(rmse_ols, rmse_glm)
)

# C. Residual Plots (The most telling comparison)
cat("Fitting log-transformed OLS for residual comparison...\n")
ols_log_model <- lm(log(fare_amount) ~ trip_distance + passenger_count,
                    data = taxi_data_clean)

cat("Generating residual plots from a 100k row sample...\n")
model_data_sample <- taxi_data_clean %>%
  sample_n(min(nrow(.), 100000))

# 1. Augment for OLS
aug_ols <- broom::augment(ols_model,
                          newdata = model_data_sample,
                          type.predict = "response",
                          type.residuals = "deviance") %>%  # <-- This argument is required
  select(.fitted_ols = .fitted, .resid_ols = .resid)

# 2. Augment for Gamma GLM
aug_glm <- broom::augment(glm_gamma_model, 
                          newdata = model_data_sample, 
                          type.predict = "response", 
                          type.residuals = "deviance") %>%  # <-- This argument is also required
  select(.fitted_glm_gamma = .fitted, .resid_glm_gamma = .resid)

# 3. Augment for Log-Transformed OLS (This is an 'lm' object, so it's fine)
aug_ols_log <- broom::augment(ols_log_model, newdata = model_data_sample) %>%
  select(.fitted_ols_log = .fitted, .resid_ols_log = .resid)

# 4. Bind all columns together
model_data <- bind_cols(model_data_sample, aug_ols, aug_glm, aug_ols_log)

# (The rest of your plotting code will now work)
cat("Plotting residuals...\n")

p_ols <- ggplot(model_data, aes(x = .fitted_ols, y = .resid_ols)) +
  geom_point(alpha = 0.2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  scale_x_continuous(labels = dollar_format()) + 
  labs(
    title = "OLS (Gaussian) Residuals",
    subtitle = "Clear heteroscedasticity (fanning pattern). Violates assumptions.",
    x = "Fitted Fare Amount ($)", y = "Deviance Residuals"
  ) +
  theme_minimal()

p_glm <- ggplot(model_data, aes(x = .fitted_glm_gamma, y = .resid_glm_gamma)) +
  geom_point(alpha = 0.2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  scale_x_continuous(labels = dollar_format()) + 
  labs(
    title = "Gamma GLM (Log Link) Residuals",
    subtitle = "Much better. Variance is more stable across fitted values.",
    x = "Fitted Fare Amount ($)", y = "Deviance Residuals"
  ) +
  theme_minimal()

p_ols_log <- ggplot(model_data, aes(x = .fitted_ols_log, y = .resid_ols_log)) +
  geom_point(alpha = 0.2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(
    title = "Log-Transformed OLS Residuals",
    subtitle = "Also a good alternative. (Fitted values are on log scale).",
    x = "Fitted log(Fare Amount)", y = "Deviance Residuals"
  ) +
  theme_minimal()

residual_plots <- (p_ols / p_glm / p_ols_log)
ggsave("plots/02_residual_comparison_plots.png", residual_plots, width = 8, height = 12)

cat("Saved residual plots to 'plots/02_residual_comparison_plots.png'.\n")

# --- 7. COEFFICIENT INTERPRETATION (GAMMA GLM) ---
coef_df <- as.data.frame(coef(summary(glm_gamma_model))) %>%
  mutate(
    Term = rownames(.),
    Multiplicative_Effect = exp(Estimate)
  ) %>%
  select(Term, Estimate, Multiplicative_Effect, `Std. Error`, `t value`, `Pr(>|t|)`)

beta_dist <- coef_df$Multiplicative_Effect[coef_df$Term == "trip_distance"]