# ---
# Title: "Practice 7: Generalized Linear Models (GLM)"
# Author: Oleksandr Severhin
# Date: November, 2025
#
# Desc: This script extends the analysis beyond OLS by fitting a
#       Gamma GLM (log link) to the NYC taxi dataset. It compares
#       performance (AIC, RMSE, Residuals) against a baseline OLS model
#       and interprets the multiplicative coefficients.
# ---

## 0. SETUP
if (!require("pacman")) install.packages("pacman")

pacman::p_load(
  arrow,       # Reading parquet
  dplyr,       # Data manipulation
  ggplot2,     # Plotting
  broom,       # Tidy output
  scales,      # Formatting
  performance, # Model metrics (Pseudo-R2)
  patchwork    # Plot combination
)

# Create directories
if (!dir.exists("data")) dir.create("data")
if (!dir.exists("plots")) dir.create("plots")

set.seed(123)

## 1. LOAD AND PREPARE DATA
file_path <- "data/cleaned_yellow_tripdata_2025-01.parquet"

if (file.exists(file_path)) {
  taxi_data <- read_parquet(file_path)
  
  # GLM Requirement: Gamma requires strictly positive response values (y > 0).
  # We also remove 0 distance trips.
  taxi_data_clean <- taxi_data %>%
    filter(fare_amount > 0, trip_distance > 0)
  
  # Sample for efficiency (GLM convergence can be slow on millions of rows)
  taxi_sample <- taxi_data_clean %>%
    sample_n(min(nrow(.), 100000))
  
  cat(sprintf("Modeling with %d rows (Filtered for positive fares).\n", nrow(taxi_sample)))
  
} else {
  stop("Dataset not found.")
}

## 2. MODEL JUSTIFICATION
# 1. Family: Gamma. 
#    Why: Fare data is continuous, strictly positive, and right-skewed.
#    Variance typically increases with the mean (heteroscedasticity), which Gamma handles.
# 2. Link: Log.
#    Why: Ensures predictions are always positive. Converts additive effects 
#    into multiplicative effects (percentages), which makes sense for pricing.

## 3. MODEL FITTING

# A. Baseline OLS (Gaussian Family, Identity Link)
# We use glm() syntax here to make AIC/Deviance directly comparable
ols_model <- glm(fare_amount ~ trip_distance + passenger_count,
                 family = gaussian(link = "identity"),
                 data = taxi_sample)

# B. Gamma GLM (Gamma Family, Log Link)
glm_gamma <- glm(fare_amount ~ trip_distance + passenger_count,
                 family = Gamma(link = "log"),
                 data = taxi_sample)

## 4. MODEL COMPARISON

# A. Metrics (AIC, Deviance, RMSE)
# Calculate RMSE manually on the training sample
preds_ols <- predict(ols_model, type = "response")
preds_glm <- predict(glm_gamma, type = "response")

rmse_ols <- sqrt(mean((taxi_sample$fare_amount - preds_ols)^2))
rmse_glm <- sqrt(mean((taxi_sample$fare_amount - preds_glm)^2))

# Compile stats
comparison_df <- data.frame(
  Model = c("OLS (Gaussian)", "GLM (Gamma)"),
  AIC = c(AIC(ols_model), AIC(glm_gamma)),
  Deviance = c(deviance(ols_model), deviance(glm_gamma)),
  RMSE = c(rmse_ols, rmse_glm),
  Pseudo_R2 = c(r2_nagelkerke(ols_model), r2_nagelkerke(glm_gamma))
)

cat("\n--- Model Performance Comparison ---\n")
print(comparison_df)

# B. Residual Diagnostics
# We compare how residuals behave. OLS usually fails here (fanning pattern).
plot_data <- data.frame(
  Fitted_OLS = fitted(ols_model),
  Resid_OLS = residuals(ols_model, type = "deviance"),
  Fitted_GLM = fitted(glm_gamma, type = "response"),
  Resid_GLM = residuals(glm_gamma, type = "deviance")
)

# Plot 1: OLS Residuals
p1 <- ggplot(plot_data, aes(x = Fitted_OLS, y = Resid_OLS)) +
  geom_point(alpha = 0.1) +
  geom_hline(yintercept = 0, col = "red", linetype = "dashed") +
  labs(title = "OLS Residuals", subtitle = "Note: Heteroscedasticity (fanning)", 
       x = "Fitted Values", y = "Deviance Residuals") +
  theme_minimal()

# Plot 2: Gamma Residuals
p2 <- ggplot(plot_data, aes(x = Fitted_GLM, y = Resid_GLM)) +
  geom_point(alpha = 0.1) +
  geom_hline(yintercept = 0, col = "red", linetype = "dashed") +
  labs(title = "Gamma GLM Residuals", subtitle = "Better variance stability", 
       x = "Fitted Values", y = "Deviance Residuals") +
  theme_minimal()

# Combine and Save
residual_plot <- p1 / p2
ggsave("plots/07_glm_residual_comparison.png", residual_plot, width = 8, height = 8)


## 5. COEFFICIENT INTERPRETATION
# Since we used link="log", coefficients are on the log scale.
# We must exponentiate them to interpret as Multiplicative Effects.

raw_coefs <- tidy(glm_gamma)

interpretation_df <- raw_coefs %>%
  mutate(
    Multiplicative_Effect = exp(estimate),
    Percent_Change = (exp(estimate) - 1) * 100
  ) %>%
  select(term, estimate, Multiplicative_Effect, Percent_Change, p.value)

cat("\n--- GLM Coefficient Interpretation (Log Link) ---\n")
print(interpretation_df)

cat("\n--- INTERPRETATION GUIDE ---\n")
cat("1. AIC/Deviance: Lower is better. Gamma should be significantly lower.\n")
cat("2. Residuals: Gamma residuals should look like a shapeless cloud. OLS usually fans out.\n")
cat("3. Coefficients (Multiplicative):\n")
cat("   - If 'trip_distance' effect is 1.25, it means a 1-unit increase in distance\n")
cat("     multiplies the fare by 1.25 (a 25% increase), holding other vars constant.\n")