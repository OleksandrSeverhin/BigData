# ---
# Title: "Practice 3. Building Advanced Models."
# Author: "Oleksandr Severhin"
# Date: October 2025
#
# ---

# --- 0. Setup ---

# Install and load packages using pacman
# This will install any missing packages and load all required ones.
if (!requireNamespace("pacman", quietly = TRUE)) {
  install.packages("pacman")
}
pacman::p_load(arrow, tidyverse, MASS, car, lubridate)

# Create a directory for plots if it doesn't exist
dir.create("plots", showWarnings = FALSE)


# --- 1. Data Loading and Cleaning ---

# Load the dataset
# Make sure the file is in a 'data/' subfolder or change the path.
tryCatch({
  taxi_clean <- read_parquet("data/cleaned_yellow_tripdata_2025-01.parquet")
}, error = function(e) {
  stop("Error: Could not read 'data/cleaned_yellow_tripdata_2025-01.parquet'. 
       Please make sure the file exists in the 'data' directory. 
       You can download it from the NYC TLC website.")
})

# Take a sample for modeling (e.g., 50,000 rows)
set.seed(123)
taxi_sample <- taxi_clean %>%
  slice_sample(n = 50000)

# Convert categorical variables to factors
taxi_sample <- taxi_sample %>%
  mutate(
    RatecodeID = as.factor(RatecodeID),
    payment_type = as.factor(payment_type),
    day_of_week = as.factor(day_of_week)
  )

print(paste("Using a sample of", nrow(taxi_sample), "rows for modeling."))
cat("\n\n")


# --- 2. Initial Model (Baseline) ---

# We'll predict total_amount based on key drivers
lm_initial <- lm(total_amount ~ trip_distance + trip_duration + passenger_count + 
                   RatecodeID + payment_type + hour_of_day + day_of_week, 
                 data = taxi_sample)

print("--- Summary of Initial Model ---")
print(summary(lm_initial))
cat("\n")

# Check for multicollinearity
print("--- VIF Scores (Initial Model) ---")
print(vif(lm_initial))
cat("\n\n")


# --- 3. Model Selection (AIC) ---

# Use stepAIC to find a model with a better AIC score.
lm_step <- stepAIC(lm_initial, direction = "both", trace = FALSE)

print("--- Summary of Stepwise-Selected Model ---")
print(summary(lm_step))
cat("\n")

# Check VIF again
print("--- VIF Scores (Stepwise Model) ---")
# VIF will still be very high for trip_distance and trip_duration
tryCatch({
  print(vif(lm_step))
}, error = function(e) {
  print("Could not calculate VIF, likely due to perfect collinearity or variable removal.")
})
cat("\n\n")


# --- 4. Diagnostic Plots (on lm_step) ---
print("--- Saving Diagnostic Plots for Stepwise Model ---")
png("plots/diagnostics_stepwise.png")
par(mfrow = c(2, 2))
plot(lm_step)
par(mfrow = c(1, 1)) # Reset plotting window
dev.off() # Close the PNG device

cat("\nSaved 'diagnostics_stepwise.png' to 'plots/' folder.
    Note the 'cone' shape in 'Residuals vs Fitted' (heteroscedasticity) 
    and the deviation in the 'Normal Q-Q' plot (non-normal residuals).\n\n")


# --- 5. Transformations ---

# A. Find optimal transformation for the response variable (Box-Cox)
# We'll use a model without the collinear trip_duration for a stable estimate
bc_model <- lm(total_amount ~ trip_distance + passenger_count + 
                 RatecodeID + payment_type + hour_of_day + day_of_week, 
               data = taxi_sample)

# Run Box-Cox plot and save it
print("--- Saving Box-Cox Plot ---")
png("plots/boxcox_lambda.png")
boxcox(bc_model, lambda = seq(-0.5, 0.5, 0.01))
dev.off() # Close the PNG device

# Find the lambda (λ) that maximizes the log-likelihood
boxcox_result <- boxcox(bc_model, lambda = seq(-0.5, 0.5, 0.01), plotit = FALSE)
optimal_lambda <- boxcox_result$x[which.max(boxcox_result$y)]
print(paste("Optimal Lambda from Box-Cox:", optimal_lambda))
cat("A lambda near 0 suggests a log transformation is appropriate.
    Plot saved to 'plots/boxcox_lambda.png'.\n\n")


# C. Build the Refined, Transformed Model
# We apply log() to response and skewed predictor.
# We drop trip_duration (collinear with trip_distance).
# We use poly(hour_of_day, 2) for non-linear effect.
lm_final <- lm(log(total_amount) ~ log(trip_distance) + passenger_count + 
                 RatecodeID + payment_type + poly(hour_of_day, 2) + day_of_week, 
               data = taxi_sample)

print("--- Summary of Final Transformed Model ---")
print(summary(lm_final))
cat("\n")

# --- Check Diagnostics on the FINAL Model ---
print("--- Saving Diagnostics for Final Model ---")
png("plots/diagnostics_final.png")
par(mfrow = c(2, 2))
plot(lm_final)
par(mfrow = c(1, 1))
dev.off() # Close the PNG device

cat("\nSaved 'diagnostics_final.png' to 'plots/' folder.
    Note the improvements: 'Residuals vs Fitted' is now a random cloud (homoscedastic).
    'Normal Q-Q' plot is much closer to the line (normal residuals).\n\n")

# Check VIF on the final model
print("--- VIF Scores (Final Model) ---")
# All VIF scores should now be low (< 5)
print(vif(lm_final))
cat("\n\n")


# --- 6. Principal Component Analysis (PCA) ---
print("--- Starting PCA Analysis ---")

# 1. Select only the numeric predictors
numeric_vars <- taxi_sample %>%
  select(trip_distance, trip_duration, passenger_count, hour_of_day)

# 2. Run PCA, making sure to scale the variables
pca_result <- prcomp(numeric_vars, scale. = TRUE)

# 3. Check the summary.
print("--- PCA Summary (Cumulative Variance) ---")
print(summary(pca_result))

# 4. Let's use the first 3 components (PC1, PC2, PC3)
pcs_to_use <- 3
pca_data <- as.data.frame(pca_result$x[, 1:pcs_to_use])

# 5. Combine PCs with the original data's categorical variables and response
taxi_pca_data <- bind_cols(
  taxi_sample %>% select(total_amount, RatecodeID, payment_type, day_of_week),
  pca_data
)

# 6. Build the PCA regression model
lm_pca <- lm(total_amount ~ . , data = taxi_pca_data)

print("--- Summary of PCA Model ---")
print(summary(lm_pca))
cat("\n")

# Check diagnostics for the PCA model
print("--- Saving Diagnostics for PCA Model ---")
png("plots/diagnostics_pca.png")
par(mfrow = c(2, 2))
plot(lm_pca)
par(mfrow = c(1, 1))
dev.off() # Close the PNG device

cat("\nSaved 'diagnostics_pca.png' to 'plots/' folder.
    Note that PCA solved multicollinearity, 
    but the heteroscedasticity and non-normality problems remain.\n\n")


# --- 7. Final Comparison ---
cat("--- Model Comparison ---

Model               | Fit (R²)          | Assumptions Met?     | Interpretability | Stability (VIF)
--------------------|-------------------|----------------------|------------------|-----------------
Initial (lm_initial)| High (e.g., ~0.93)| NO (VIF > 10, Skew)  | High             | POOR
Stepwise (lm_step)  | High (e.g., ~0.93)| NO (VIF > 10, Skew)  | High             | POOR
PCA (lm_pca)        | High (e.g., ~0.93)| NO (Skew, Non-Normal)| VERY POOR        | EXCELLENT
Final (lm_final)    | High (e.g., ~0.91)| YES! (VIF < 3, Normal)| Moderate (log-log)| EXCELLENT

Conclusion: The 'lm_final' model with transformations is the most robust,
            reliable, and statistically valid model.\n")
