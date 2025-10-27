# ---
# Title: "Practice 3. Building Advanced Models."
# Author: "Oleksandr Severhin"
# Date: October 2025
#
# ---

# --- 0. Setup ---
if (!requireNamespace("pacman", quietly = TRUE)) {
  install.packages("pacman")
}
pacman::p_load(arrow, tidyverse, MASS, car, lubridate)

# --- 1. Data Loading and Cleaning ---
tryCatch({
  taxi_data <- read_parquet("data/cleaned_yellow_tripdata_2025-01.parquet")
}, error = function(e) {
  stop("Error: Could not read 'data/cleaned_yellow_tripdata_2025-01.parquet'. 
       Please make sure the file exists in the 'data' directory.")
})

taxi_sample <- taxi_data

taxi_sample <- taxi_sample %>%
  mutate(
    payment_method = as.factor(payment_method),
    day_of_week = as.factor(day_of_week)
  )

print(paste("Using a sample of", nrow(taxi_sample), "rows for modeling."))

# --- 2. Initial Model (Baseline) ---
lm_initial <- lm(total_amount ~ trip_distance + passenger_count + payment_method + pickup_hour + day_of_week,
  data = taxi_sample
)

print("--- Summary of Initial Model ---")
print(summary(lm_initial))

# Check for multicollinearity
print("--- VIF Scores (Initial Model) ---")
print(vif(lm_initial))

# --- 3. Model Selection (AIC) ---
lm_step <- stepAIC(lm_initial, direction = "both", trace = FALSE)

print("--- Summary of Stepwise-Selected Model ---")
print(summary(lm_step))

print("--- VIF Scores (Stepwise Model) ---")
tryCatch({
  print(vif(lm_step))
}, error = function(e) {
  print("Could not calculate VIF, likely due to perfect collinearity or variable removal.")
})

# --- 4. Diagnostic Plots (on lm_step) ---
print("--- Saving Diagnostic Plots for Stepwise Model ---")
png("plots/diagnostics_stepwise.png")
par(mfrow = c(2, 2))
plot(lm_step)
par(mfrow = c(1, 1)) 
dev.off() 

cat("\nSaved 'diagnostics_stepwise.png' to 'plots/' folder.
    Note the 'cone' shape in 'Residuals vs Fitted' (heteroscedasticity) 
    and the deviation in the 'Normal Q-Q' plot (non-normal residuals).\n\n")


# --- 5. Transformations ---
# A. Find optimal transformation for the response variable (Box-Cox)
bc_model <- lm(total_amount ~ trip_distance + passenger_count + 
                 payment_method + pickup_hour + day_of_week, 
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
lm_final <- lm(log(total_amount) ~ log(trip_distance) + passenger_count + 
                 payment_method + poly(pickup_hour, 2) + day_of_week, 
               data = taxi_sample)

print("--- Summary of Final Transformed Model ---")
print(summary(lm_final))

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


# --- 6. Principal Component Analysis (PCA) ---
print("--- Starting PCA Analysis ---")

# 1. Select only the numeric predictors from your new model
numeric_vars <- taxi_sample %>%
  dplyr::select(trip_distance, passenger_count, pickup_hour)

# 2. Run PCA, making sure to scale the variables
pca_result <- prcomp(numeric_vars, scale. = TRUE)

# 3. Check the summary.
print("--- PCA Summary (Cumulative Variance) ---")
print(summary(pca_result))

# 4. Let's use all 3 components (or adjust 'pcs_to_use' if needed)
pcs_to_use <- 3
pca_data <- as.data.frame(pca_result$x[, 1:pcs_to_use])

# 5. Combine PCs with the original data's categorical variables and response
taxi_pca_data <- bind_cols(
   taxi_sample %>% dplyr::select(total_amount, payment_method, day_of_week), # Use dplyr::select
   pca_data
)

# 6. Build the PCA regression model
lm_pca <- lm(total_amount ~ . , data = taxi_pca_data)

print("--- Summary of PCA Model ---")
print(summary(lm_pca))

# Check diagnostics for the PCA model
print("--- Saving Diagnostics for PCA Model ---")
png("plots/diagnostics_pca.png")
par(mfrow = c(2, 2))
plot(lm_pca)
par(mfrow = c(1, 1))
dev.off()

cat("\nSaved 'diagnostics_pca.png' to 'plots/' folder.
    Note that PCA solved multicollinearity (if any), 
    but the heteroscedasticity and non-normality problems remain.\n\n")
