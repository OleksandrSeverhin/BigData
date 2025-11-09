# ---
# Title: "Practice 4: Full Model Refinement (Full Dataset)"
# Author: Oleksandr Severhin
# Date: October 31, 2025
#
# Desc: This script runs the advanced model refinement on the
#       ENTIRE dataset. It uses log(x+1) to avoid filtering 0 values.
#       It fits all models and runs all diagnostics on the full dataset.
# ---

# --- 0. SETUP: LOAD LIBRARIES AND DATA ---

if (!requireNamespace("pacman", quietly = TRUE)) {
  install.packages("pacman")
}
# Load all required libraries
pacman::p_load(arrow, dplyr, ggplot2, broom, MASS, car, knitr, ggcorrplot)

Sys.setlocale("LC_TIME", "English")

# Load and prepare the data
tryCatch({
  taxi_clean <- read_parquet("data/cleaned_yellow_tripdata_2025-01.parquet")
  taxi_model_data <- taxi_clean %>%
    mutate(
      log_total = log(total_amount + 1),
      log_distance = log(trip_distance + 1),
      log_fare = log(fare_amount + 1),
      log_tip = log(tip_amount + 1),
      log_tolls = log(tolls_amount + 1),
      payment_type = as.factor(payment_type),
      day_of_week = as.factor(day_of_week),
      VendorID = as.factor(VendorID)
    )
  
}, error = function(e) {
  print("--- DATA LOADING ERROR ---")
  print(e)
})


# --- 1. FULL LOG MODEL & MULTICOLLINEARITY (ON FULL DATA) ---

# This is our "baseline" model, fit on all rows.
full_log_model <- lm(log_total ~ log_distance + log_fare + log_tip + log_tolls +
                       passenger_count + payment_type + pickup_hour
                     + day_of_week + VendorID,
                     data = taxi_model_data)

# Print VIF scores
vif_scores <- vif(full_log_model)
print("VIF Scores for Full Log Model:")
print(vif_scores)

# Save correlation plot
numeric_vars <- taxi_model_data %>%
  dplyr::select(log_total, log_distance, log_fare, log_tip, log_tolls,
                passenger_count, pickup_hour)
corr <- round(cor(numeric_vars, use = "complete.obs"), 2)

png("plots/correlation_heatmap.png", width = 800, height = 700, res = 120)
print(
  ggcorrplot(corr,
             hc.order = TRUE,
             type = "lower",
             lab = TRUE,
             lab_size = 3,
             title = "Correlation of Numeric Predictors")
)
dev.off()


# --- 2. MODEL SELECTION & PCA (ON FULL DATA) ---

# --- Model A: Stepwise BIC Model ---
n_obs <- nrow(taxi_model_data)
model_bic <- stepAIC(full_log_model,
                     direction = "backward",
                     trace = FALSE,
                     k = log(n_obs)) # k = log(n) selects by BIC
cat("\n--- Summary: BIC Stepwise Model --- \n")
print(summary(model_bic))


# --- Model B: Manually Refined Model (for stability) ---
model_manual <- lm(log_total ~ log_distance + log_tip + log_tolls +
                     passenger_count + payment_type + pickup_hour +
                     day_of_week + VendorID,
                   data = taxi_model_data)
cat("\n--- Summary: Manual Model --- \n")
print(summary(model_manual))


# --- Model C: Principal Component Analysis (PCA) Model ---
cor_vars <- taxi_model_data %>%
  dplyr::select(log_distance, log_fare, log_tip, log_tolls)

# PCA runs on the full dataset
pca_results <- prcomp(cor_vars, scale. = TRUE)

# Create the new dataset with PCA columns
pca_data <- bind_cols(
  taxi_model_data,
  as.data.frame(pca_results$x)
)

# Fit the PCA model on the full dataset
model_pcr <- lm(log_total ~ PC1 + PC2 + PC3 + PC4 +
                  passenger_count + payment_type + pickup_hour
                + day_of_week + VendorID,
                data = pca_data)
cat("\n--- Summary: PCA Model (Full) --- \n")
print(summary(model_pcr))

# Save PCA Scree Plot
png(filename = "plots/pca_scree_plot.png", width = 800, height = 600, res = 120)
plot(pca_results, type = "l", main = "PCA Scree Plot")
dev.off()


# --- 3. DIAGNOSTIC PLOTS (ON FULL DATA) ---

# Plot diagnostics for BIC model
print("Saving BIC model diagnostics...")
png(filename = "plots/diagnostics_bic_model.png", width = 1000, height = 1000, res = 120)
par(mfrow = c(2, 2))
plot(model_bic)
par(mfrow = c(1, 1))
dev.off()

# Plot diagnostics for MANUAL model
print("Saving Manual model diagnostics...")
png(filename = "plots/diagnostics_manual_model.png", width = 1000, height = 1000, res = 120)
par(mfrow = c(2, 2))
plot(model_manual)
par(mfrow = c(1, 1))
dev.off()

# Plot diagnostics for PCA model
print("Saving PCA model diagnostics...")
png(filename = "plots/diagnostics_pca_model.png", width = 1000, height = 1000, res = 120)
par(mfrow = c(2, 2))
plot(model_pcr)
par(mfrow = c(1, 1))
dev.off()


# --- 4. COMPARING THE FINAL MODELS (FIT ON FULL DATA) ---

# Extract Summaries
summary_bic <- summary(model_bic)
summary_manual <- summary(model_manual)
summary_pcr <- summary(model_pcr)

# Get VIF scores
vif_bic <- max(vif(model_bic)["log_distance"], vif(model_bic)["log_fare"])
vif_manual <- max(vif(model_manual))
vif_pcr <- max(vif(model_pcr))

# Create the final comparison table
final_stats <- data.frame(
  Model = c("BIC Stepwise", "Manual (No Fare)", "PCA (Full)"),
  Adj_R_Squared = c(
    summary_bic$adj.r.squared,
    summary_manual$adj.r.squared,
    summary_pcr$adj.r.squared
  ),
  RSE = c(
    summary_bic$sigma,
    summary_manual$sigma,
    summary_pcr$sigma
  ),
  Max_VIF = c(vif_bic, vif_manual, vif_pcr),
  Interpretability = c("Flawed/None", "High", "Very Low")
)

# Print the final comparison table
cat("\n--- Comparison Table: Refined Log-Log Models (Fit on Full Data) --- \n")
print(knitr::kable(final_stats, digits = 4))
