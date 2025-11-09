# ---
# Title: "Practice 3. Building Advanced Models."
# Author: Oleksandr Severhin
# Date: October 2025
#
# Desc: This script builds on initial models by applying advanced
#       techniques: BIC for model selection, PCA for multicollinearity,
#       and a final comparison of candidate models.
# ---

# --- 0. Setup ---
if (!requireNamespace("pacman", quietly = TRUE)) {
  install.packages("pacman")
}
pacman::p_load(arrow, tidyverse, MASS, car, lubridate, knitr, broom)

# Set locale for consistent day names
Sys.setlocale("LC_TIME", "English")

# --- 1. Data Loading and Transformation ---
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

print(paste("Loaded and transformed", nrow(taxi_model_data), "rows."))


# --- 2. Initial "Full" Model and Multicollinearity ---
print("--- Fitting 'Full' Model for VIF Check ---")
full_model <- lm(log_total ~ log_distance + log_fare + log_tip + log_tolls + 
                   passenger_count + payment_type + pickup_hour + day_of_week + VendorID,
                 data = taxi_model_data)

# Check for multicollinearity
vif_scores <- vif(full_model)
print("--- VIF Scores (Full Model) ---")
print(vif_scores)
cat("\nNote: High VIF scores for log_distance and log_fare indicate multicollinearity.\n\n")

# --- 3. Model Selection (BIC) ---
# Determine n (number of observations) for BIC calculation
n_obs <- nrow(taxi_model_data)

print("--- Running Backward Stepwise Selection (BIC) ---")
# Use BIC (k = log(n_obs)) instead of AIC for a stricter penalty
step_model_bic <- stepAIC(full_model, 
                          direction = "backward", 
                          trace = FALSE, 
                          k = log(n_obs)) 

print("--- Summary of Stepwise-Selected Model (BIC) ---")
print(summary(step_model_bic))
cat("\nNote: BIC model is simpler but may still contain collinear variables.\n\n")


# --- 4. Advanced Diagnostics (on BIC-Selected Model) ---
print("--- Generating Diagnostics for BIC Model ---")
set.seed(123)
diag_sample <- taxi_model_data %>% slice_sample(n = 10000)

# Re-fit the BIC-selected model formula on the sample
step_model_diag <- lm(formula(step_model_bic), data = diag_sample)

# Save plots to the 'plots' folder
png("plots/diagnostics_step_bic.png", width = 800, height = 800)
par(mfrow = c(2, 2))
plot(step_model_diag, which = 1) # Residuals vs Fitted
plot(step_model_diag, which = 2) # Normal Q-Q
plot(step_model_diag, which = 4) # Cook's Distance
plot(step_model_diag, which = 5) # Residuals vs Leverage
par(mfrow = c(1, 1))
dev.off()

cat("Saved 'diagnostics_step_bic.png' to 'plots/' folder.\n\n")

# --- 5. Model Improvement: Principal Component Analysis (PCA) ---
print("--- Starting PCA Analysis on Correlated Variables ---")

# 1. Isolate ONLY the correlated numeric predictors
cor_vars <- taxi_model_data %>%
  dplyr::select(log_distance, log_fare, log_tip, log_tolls)

# 2. Run PCA, making sure to scale the variables
pca_results <- prcomp(cor_vars, scale. = TRUE)

print("--- PCA Summary (Cumulative Variance) ---")
print(summary(pca_results))

# 3. Create the PCA dataset
pca_data <- bind_cols(
  # Select all variables *except* the ones replaced by PCA
  taxi_model_data %>%
    dplyr::select(-log_distance, -log_fare, -log_tip, -log_tolls),
  # Add the new PC components
  as.data.frame(pca_results$x)
)

# 4. Fit PCA model (Full: all 4 PCs + original non-numeric)
# This model uses the correct log_total response
pca_model_full <- lm(log_total ~ PC1 + PC2 + PC3 + PC4 +
                     passenger_count + payment_type + pickup_hour + day_of_week + VendorID,
                     data = pca_data)

# 5. Fit PCA model (Reduced: first 3 PCs, no VendorID)
pca_model_reduced <- lm(log_total ~ PC1 + PC2 + PC3 +
                        passenger_count + payment_type + pickup_hour + day_of_week,
                        data = pca_data)

print("--- Summary of Full PCA Model ---")
print(summary(pca_model_full))

# --- 6. Final Model Comparison ---
print("--- Generating Final Model Comparison Table ---")

# --- Function to extract F-statistic p-value ---
get_f_pvalue <- function(model_summary) {
  fstat <- model_summary$fstatistic
  if (is.null(fstat) || length(fstat) < 3) return(NA)
  pvalue <- pf(fstat[1], fstat[2], fstat[3], lower.tail = FALSE)
  return(pvalue)
}

# --- Extract Summaries ---
summary_bic <- summary(step_model_bic)
summary_pca_full <- summary(pca_model_full)
summary_pca_reduced <- summary(pca_model_reduced)

# --- Create Expanded Data Frame ---
model_stats_f <- data.frame(
  Model = c("Stepwise (BIC)", "PCA (Full)", "PCA (Reduced)"),
  Num_Predictors = c(
    length(coef(step_model_bic)) - 1, 
    length(coef(pca_model_full)) - 1,
    length(coef(pca_model_reduced)) - 1
  ),
  Adj_R_Squared = c(
    summary_bic$adj.r.squared,
    summary_pca_full$adj.r.squared,
    summary_pca_reduced$adj.r.squared
  ),
  RSE = c( # Residual Standard Error
    summary_bic$sigma,
    summary_pca_full$sigma,
    summary_pca_reduced$sigma
  ),
  F_Statistic = c(
    summary_bic$fstatistic[1], # Value of F
    summary_pca_full$fstatistic[1],
    summary_pca_reduced$fstatistic[1]
  ),
  F_p_value_raw = c(
    get_f_pvalue(summary_bic),
    get_f_pvalue(summary_pca_full),
    get_f_pvalue(summary_pca_reduced)
  ),
  AIC = c(AIC(step_model_bic), AIC(pca_model_full), AIC(pca_model_reduced)),
  BIC = c(BIC(step_model_bic), BIC(pca_model_full), BIC(pca_model_reduced))
)

# Format p-value nicely
model_stats_f$F_p_value <- ifelse(model_stats_f$F_p_value_raw < 0.001, "< 0.001", round(model_stats_f$F_p_value_raw, 3))
model_stats_f$F_p_value_raw <- NULL # Remove the raw column

# Reorder columns for clarity
model_stats_f <- model_stats_f[, c("Model", "Num_Predictors", "Adj_R_Squared", "RSE", "F_Statistic", "F_p_value", "AIC", "BIC")]

# Print the final kable table to the console
print(kable(model_stats_f,
            caption = "Comparison of Final Models (BIC vs. PCA)",
            digits = c(NA, 0, 4, 4, 0, NA, 0, 0),
            col.names = c("Model", "# Predictors", "Adj. R²", "RSE", "F-statistic", "F p-value", "AIC", "BIC")))
