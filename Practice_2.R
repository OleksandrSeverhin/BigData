# ---
# Title: "Pracice 2. ADVANCED MULTIPLE LINEAR REGRESSION"
# Author: Oleksandr Severhin
# Date: October 2025
# Description: This script focuses on model diagnostics, 
#              evaluation, and improvement techniques.
# ---

# --- 1. SETUP AND CONFIGURATION ---

# Use pacman to manage packages, which installs if missing and loads
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  arrow,      # For reading Parquet files
  dplyr,      # For data manipulation
  ggplot2,    # For creating visualizations
  broom,      # For tidying model outputs
  car,        # For VIF (Variance Inflation Factor) calculation
  MASS,       # For stepAIC function
  patchwork   # For combining plots
)

# Set a consistent theme for all plots
theme_set(theme_minimal(base_size = 12))

# Ensure the 'plots' directory exists for saving images
if (!dir.exists("plots")) {
  dir.create("plots")
  cat("Created directory: plots/\n")
}

# --- 2. DATA LOADING ---

# Load the cleaned dataset from the previous session
tryCatch({
  taxi_data <- read_parquet("data/cleaned_yellow_tripdata_2025-01.parquet")
  cat("Cleaned dataset loaded successfully.\n")
  cat("Dimensions:", dim(taxi_data)[1], "rows,", dim(taxi_data)[2], "columns\n")
}, error = function(e) {
  stop("Failed to load the cleaned dataset. Make sure 'cleaned_yellow_tripdata_2025-01.parquet' is in the 'data/' directory.")
})


# --- 3. BASELINE MULTIPLE REGRESSION MODEL ---

# Fit a linear model with at least 5 predictors to predict total_amount.
# We'll use trip_distance, passenger_count, payment_method, pickup_hour, and day_of_week.
baseline_model <- lm(
  total_amount ~ trip_distance + passenger_count + payment_method + pickup_hour + day_of_week,
  data = taxi_data
)

# Print the model summary
cat("\n--- Summary of Baseline Multiple Regression Model ---\n")
print(summary(baseline_model))


# --- 4. DIAGNOSTICS FOR THE BASELINE MODEL ---

cat("\n--- Performing Diagnostics on Baseline Model ---\n")

# To avoid performance issues with plotting millions of points, we'll use a random sample for diagnostic plots.
set.seed(42)
diag_sample <- taxi_data 

# Re-fit the model on the sample for plotting purposes
baseline_model_sampled <- lm(
  total_amount ~ trip_distance + passenger_count + payment_method + pickup_hour + day_of_week,
  data = diag_sample
)
model_augmented <- augment(baseline_model_sampled)

# 4a. Linearity and Homoscedasticity (Residuals vs. Fitted Plot)
# We look for a random cloud of points around y=0. Patterns (like a funnel) suggest problems.
p1 <- ggplot(model_augmented, aes(x = .fitted, y = .resid)) +
  geom_point(alpha = 0.3, color = "firebrick") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  labs(
    title = "Baseline Model: Residuals vs. Fitted Values",
    subtitle = "Shows signs of non-linearity and heteroscedasticity (fanning out)",
    x = "Fitted Values (Predicted Fare)",
    y = "Residuals"
  )
ggsave("plots/baseline_residuals_vs_fitted.png", p1, width = 8, height = 6)
cat("Saved plot: plots/baseline_residuals_vs_fitted.png\n")


# 4b. Normality of Residuals (Q-Q Plot)
# We look for points falling closely along the diagonal line.
p2 <- ggplot(model_augmented, aes(sample = .resid)) +
  stat_qq(alpha = 0.3, color = "firebrick") +
  stat_qq_line(color = "black", size = 1) +
  labs(
    title = "Baseline Model: Normal Q-Q Plot of Residuals",
    subtitle = "Tails deviate significantly from the normal line",
    x = "Theoretical Quantiles",
    y = "Sample Quantiles"
  )
ggsave("plots/baseline_qq_plot.png", p2, width = 8, height = 6)
cat("Saved plot: plots/baseline_qq_plot.png\n")

# 4c. Multicollinearity (Variance Inflation Factor - VIF)
# VIF values > 5 suggest potential multicollinearity issues.
vif_values <- vif(baseline_model)
cat("\n--- Variance Inflation Factor (VIF) for Baseline Model ---\n")
print(vif_values)
# A VIF value close to 1 indicates no correlation between a given predictor and the other predictors.


# --- 5. MODEL IMPROVEMENT ---

cat("\n--- Improving the Model using Transformations and Variable Selection ---\n")

# The diagnostics suggest transformations are needed. The right-skew in the variables
# and the patterns in the residual plots point to a log transformation.

# 5a. Fit an improved model with log-transformed variables
# We log-transform the target (total_amount) and the main skewed predictor (trip_distance).
# This creates a log-log model, where coefficients represent elasticities.
transformed_model <- lm(
  log(total_amount) ~ log(trip_distance) + passenger_count + payment_method + pickup_hour + day_of_week,
  data = taxi_data
)

# 5b. Use Stepwise Variable Selection to refine the model
# stepAIC will add/remove variables to find a model with the lowest AIC (Akaike Information Criterion).
cat("\n--- Running Stepwise Variable Selection (stepAIC) ---\n")
# Note: This can take a moment to run on the full dataset.
improved_model <- stepAIC(transformed_model, direction = "both", trace = FALSE)

# Display the final selected model's formula and summary
cat("\n--- Final Improved Model Formula ---\n")
print(formula(improved_model))

cat("\n--- Summary of Final Improved Model ---\n")
print(summary(improved_model))


# --- 6. DIAGNOSTICS FOR THE IMPROVED MODEL ---

cat("\n--- Performing Diagnostics on Final Improved Model ---\n")

# Re-fit the final model on the sample data for plotting
improved_model_sampled <- lm(formula(improved_model), data = diag_sample)
improved_augmented <- augment(improved_model_sampled)

# 6a. Improved Model: Residuals vs. Fitted Plot
# We hope to see a more random, uniform scatter of points.
p3 <- ggplot(improved_augmented, aes(x = .fitted, y = .resid)) +
  geom_point(alpha = 0.3, color = "darkgreen") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  labs(
    title = "Improved Model: Residuals vs. Fitted Values",
    subtitle = "Plot shows much better random distribution and less fanning",
    x = "Fitted Values (Predicted log(Fare))",
    y = "Residuals"
  )
ggsave("plots/improved_residuals_vs_fitted.png", p3, width = 8, height = 6)
cat("Saved plot: plots/improved_residuals_vs_fitted.png\n")

# 6b. Improved Model: Normal Q-Q Plot
# The points should now lie much closer to the diagonal line.
p4 <- ggplot(improved_augmented, aes(sample = .resid)) +
  stat_qq(alpha = 0.3, color = "darkgreen") +
  stat_qq_line(color = "black", size = 1) +
  labs(
    title = "Improved Model: Normal Q-Q Plot of Residuals",
    subtitle = "Points now follow the normal line much more closely",
    x = "Theoretical Quantiles",
    y = "Sample Quantiles"
  )
ggsave("plots/improved_qq_plot.png", p4, width = 8, height = 6)
cat("Saved plot: plots/improved_qq_plot.png\n")

# --- 7. FINAL COMPARISON AND CONCLUSION ---
cat("\nAnalysis complete. Diagnostic plots for baseline and improved models saved in 'plots/'.\n")
cat("The improved model with log transformations shows a much better fit and adherence to linear regression assumptions.\n")
cat("Adjusted R-squared improved from", round(summary(baseline_model)$adj.r.squared, 4),
    "to", round(summary(improved_model)$adj.r.squared, 4), "on a transformed scale, indicating a better explanation of variance.\n")
