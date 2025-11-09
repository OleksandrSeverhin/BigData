# ---
# Title: "Practice 5: Full Model Refinement (Full Dataset)"
# Author: Oleksandr Severhin
# Date: November, 2025
#
# Desc: This script runs the advanced model refinement on the
#       ENTIRE dataset. It uses log(x+1) to avoid filtering 0 values.
#       It fits all models and runs all diagnostics on the full dataset.
# ---

## 0. SETUP 
# Install pacman if it's not already installed
if (!require("pacman")) install.packages("pacman")

# Load libraries using pacman::p_load
pacman::p_load(
  arrow,       # For reading parquet files
  dplyr,       # For data manipulation
  ggplot2,     # For plotting
  broom,       # For tidying model output
  scales,      # For plot formatting
  performance, # For pseudo-R2
  patchwork      # For combining plots
)

# Ensure 'data' and 'plots' directories exist
if (!dir.exists("data")) dir.create("data")
if (!dir.exists("plots")) dir.create("plots")

# Set seed for reproducibility
set.seed(123)

## 1. LOAD AND PREPARE DATA
file_path <- "data/cleaned_yellow_tripdata_2025-01.parquet"

if (file.exists(file_path)) {
  taxi_data <- read_parquet(file_path)
  
  cat(sprintf("Loaded %d rows from the full dataset.\n", nrow(taxi_data)))
  taxi_data_clean <- taxi_data %>%
    filter(fare_amount > 0, trip_distance > 0)
  
  cat(sprintf("Filtered to %d rows with fare_amount > 0 and trip_distance > 0 for modeling.\n", nrow(taxi_data_clean)))
  
} else {
  stop(sprintf("Error: Dataset not found at %s. Please check the file path.", file_path))
}


## 2. EXPLORATORY DATA ANALYSIS (JUSTIFICATION FOR GLM)
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


## 3. MODEL JUSTIFICATION


## 4. MODEL 1: BASELINE OLS (FIT AS A GLM FOR COMPARISON)
cat("Fitting Model 1: OLS (Gaussian GLM) on full, clean dataset...\n")
ols_model <- glm(fare_amount ~ trip_distance + passenger_count,
                 family = gaussian(link = "identity"),
                 data = taxi_data_clean)

summary(ols_model)


## 5. MODEL 2: GAMMA GLM (LOG LINK)
cat("Fitting Model 2: Gamma GLM (Log Link) on full, clean dataset...\n")
glm_gamma_model <- glm(fare_amount ~ trip_distance + passenger_count,
                       family = Gamma(link = "log"),
                       data = taxi_data_clean)

summary(glm_gamma_model)


## 6. MODEL COMPARISON
cat("\n--- MODEL COMPARISON ---\n")

# A. AIC and Deviance
model_comp_stats <- data.frame(
  Model = c("OLS (Gaussian)", "GLM (Gamma)"),
  AIC = c(AIC(ols_model), AIC(glm_gamma_model)),
  Deviance = c(deviance(ols_model), deviance(glm_gamma_model)),
  Pseudo_R2_Tjur = c(r2_tjur(ols_model), r2_tjur(glm_gamma_model)) # Good for binary, but works
)
cat("Fit Statistics:\n")
print(model_comp_stats)
cat("*Lower AIC/Deviance is better. The Gamma GLM's AIC is likely much lower, \nindicating a better fit for the data's structure.\n")


# B. RMSE (Root Mean Squared Error)
preds_ols <- predict(ols_model, newdata = taxi_data_clean, type = "response")
preds_glm <- predict(glm_gamma_model, newdata = taxi_data_clean, type = "response")

rmse_ols <- sqrt(mean((taxi_data_clean$fare_amount - preds_ols)^2))
rmse_glm <- sqrt(mean((taxi_data_clean$fare_amount - preds_glm)^2))

rmse_comp <- data.frame(
  Model = c("OLS (Gaussian)", "GLM (Gamma)"),
  RMSE = c(rmse_ols, rmse_glm)
)
cat("\nPredictive Accuracy (RMSE on Fare Amount):\n")
print(rmse_comp)
cat("*Lower RMSE is better. The Gamma GLM's predictions are likely more accurate.\n")


# C. Residual Plots (The most telling comparison)
cat("Fitting log-transformed OLS for residual comparison...\n")

ols_log_model <- lm(log(fare_amount) ~ trip_distance + passenger_count,
                    data = taxi_data_clean)

cat("Generating residual plots from a 100k row sample...\n")
model_data_sample <- taxi_data_clean %>%
  sample_n(min(nrow(.), 100000))

# Augment data with predictions and residuals from all 3 models
model_data <- model_data_sample %>%
  mutate(
    .fitted_ols = predict(ols_model, newdata = model_data_sample, type = "response"),
    .resid_ols = residuals(ols_model, type = "deviance")[as.numeric(rownames(model_data_sample))], # Get residuals for the sampled rows
    
    .fitted_glm_gamma = predict(glm_gamma_model, newdata = model_data_sample, type = "response"),
    .resid_glm_gamma = residuals(glm_gamma_model, type = "deviance")[as.numeric(rownames(model_data_sample))],
    
    .fitted_ols_log = predict(ols_log_model, newdata = model_data_sample, type = "response"),
    .resid_ols_log = residuals(ols_log_model, type = "deviance")[as.numeric(rownames(model_data_sample))]
  )

# Plot 1: OLS Residuals
p_ols <- ggplot(model_data, aes(x = .fitted_ols, y = .resid_ols)) +
  geom_point(alpha = 0.2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(
    title = "OLS (Gaussian) Residuals",
    subtitle = "Clear heteroscedasticity (fanning pattern). Violates assumptions.",
    x = "Fitted Fare Amount ($)", y = "Deviance Residuals"
  ) +
  theme_minimal()

# Plot 2: Gamma GLM Residuals
p_glm <- ggplot(model_data, aes(x = .fitted_glm_gamma, y = .resid_glm_gamma)) +
  geom_point(alpha = 0.2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(
    title = "Gamma GLM (Log Link) Residuals",
    subtitle = "Much better. Variance is more stable across fitted values.",
    x = "Fitted Fare Amount ($)", y = "Deviance Residuals"
  ) +
  theme_minimal()

# Plot 3: Log-Transformed OLS Residuals
p_ols_log <- ggplot(model_data, aes(x = .fitted_ols_log, y = .resid_ols_log)) +
  geom_point(alpha = 0.2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(
    title = "Log-Transformed OLS Residuals",
    subtitle = "Also a good alternative. (Fitted values are on log scale).",
    x = "Fitted log(Fare Amount)", y = "Deviance Residuals"
  ) +
  theme_minimal()

# Combine and save plots
residual_plots <- (p_ols / p_glm / p_ols_log)
ggsave("plots/02_residual_comparison_plots.png", residual_plots, width = 8, height = 12)

cat("Saved residual plots to 'plots/02_residual_comparison_plots.png'.\n")

## 7. COEFFICIENT INTERPRETATION (GAMMA GLM)
cat("\n--- GLM COEFFICIENT INTERPRETATION --- \n")
print(summary(glm_gamma_model))

# To interpret coefficients from a log link, we must exponentiate them.
# The raw coefficient (beta) is on the log-scale.
coef_df <- as.data.frame(coef(summary(glm_gamma_model))) %>%
  mutate(
    Term = rownames(.),
    Multiplicative_Effect = exp(Estimate)
  ) %>%
  select(Term, Estimate, Multiplicative_Effect, `Std. Error`, `z value`, `Pr(>|z|)`)

cat("\nExponentiated Coefficients (Multiplicative Effects):\n")
print(coef_df, digits = 8)

beta_dist <- coef_df$Multiplicative_Effect[coef_df$Term == "trip_distance"]
