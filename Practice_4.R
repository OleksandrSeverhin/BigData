# ---
# Title: "Practice 4: Advanced Model Refinement"
# Author: "Based on yehorbolt's .qmd"
# Date: October 2025
#
# Desc: This script refines a regression model using BIC for model
#       selection, PCA for multicollinearity, and in-depth diagnostics
#       to compare the final candidate models.
# ---

# --- 0. SETUP: LOAD LIBRARIES AND DATA ---
if (!requireNamespace("pacman", quietly = TRUE)) {
  install.packages("pacman")
}
pacman::p_load(arrow, dplyr, ggplot2, broom, MASS, car, knitr, ggcorrplot)

Sys.setlocale("LC_TIME", "English")

tryCatch({
  taxi_clean <- read_parquet("data/cleaned_yellow_tripdata_2025-01.parquet")
  taxi_model_data <- taxi_clean %>%
    mutate(
      log_total = log(total_amount),
      log_distance = log(trip_distance),
      log_fare = log(fare_amount),
      log_tip = log(tip_amount + 1),
      log_tolls = log(tolls_amount + 1),
      # Convert categorical variables to factors for regression
      payment_type = as.factor(payment_type),
      day_of_week = as.factor(day_of_week),
      VendorID = as.factor(VendorID)
    )
  print("Successfully loaded and transformed data.")
}, error = function(e) {
  print("--- ERROR ---")
  print("Please ensure the file exists and the path is correct.")
  print(e)
})


# --- 1. INITIAL MODEL & MULTICOLLINEARITY CHECK ---
print("--- 1. Building Full Model for VIF Check ---")

# Fit the initial "full" model to check for multicollinearity
full_model <- lm(log_total ~ log_distance + log_fare + log_tip + log_tolls +
                   passenger_count + payment_type + pickup_hour
                 + day_of_week + VendorID,
                 data = taxi_model_data)

# Print the VIF scores
vif_scores <- vif(full_model)
print("VIF Scores for Full Model (High > 5):")
print(vif_scores)

# Save correlation plot
print("Saving correlation heatmap to plots/correlation_heatmap.png")
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

# --- 2. APPLYING MODEL SELECTION (BIC, MANUAL) & PCA ---
print("--- 2. Developing 3 Candidate Models ---")

# --- Candidate A: Stepwise BIC Model ---
print("Building Model A: Stepwise BIC...")
n_obs <- nrow(taxi_model_data)
step_model_bic <- stepAIC(full_model,
                          direction = "backward",
                          trace = FALSE,
                          k = log(n_obs))
cat("\n--- Summary: BIC Stepwise Model --- \n")
print(summary(step_model_bic))

# --- Candidate B: Manually Refined Model ---
print("Building Model B: Manual (Removed log_fare)...")
manual_model <- lm(log_total ~ log_distance + log_tip + log_tolls +
                     passenger_count + payment_type + pickup_hour +
                     day_of_week + VendorID,
                   data = taxi_model_data)
cat("\n--- Summary: Manual Model --- \n")
print(summary(manual_model))
print("VIF Scores for Manual Model (Multicollinearity solved):")
print(vif(manual_model))

# --- Candidate C: Principal Component Analysis (PCA) Model ---
print("Building Model C: PCA Model...")
# 1. Isolate the correlated numeric predictors
cor_vars <- taxi_model_data %>%
  dplyr::select(log_distance, log_fare, log_tip, log_tolls)

# 2. Run PCA
pca_results <- prcomp(cor_vars, scale. = TRUE)
print(summary(pca_results))

# 3. Create a new dataset with the PCs
pca_data <- bind_cols(
  taxi_model_data %>% dplyr::select(-log_distance, -log_fare, -log_tip, -log_tolls),
  as.data.frame(pca_results$x)
)

# 4. Fit a new model using all 4 PCs
pca_model_full <- lm(log_total ~ PC1 + PC2 + PC3 + PC4 +
                       passenger_count + payment_type + pickup_hour
                     + day_of_week + VendorID,
                     data = pca_data)
cat("\n--- Summary: PCA Model (Full) --- \n")
print(summary(pca_model_full))

# 5. Save PCA Scree Plot
print("Saving PCA scree plot to plots/pca_scree_plot.png")
png(filename = "plots/pca_scree_plot.png", width = 800, height = 600, res = 120)
plot(pca_results, type = "l", main = "PCA Scree Plot")
mtext("Proportion of Variance Explained by Component")
dev.off()

# --- 3. CHECKING DIAGNOSTIC PLOTS ---
print("--- 3. Saving Diagnostic Plots (on a 10k Sample) ---")

# Create a sample for fast diagnostics
set.seed(123)
diag_sample <- taxi_model_data %>% slice_sample(n = 10000)

# Re-fit models on the sample to generate plots
step_model_diag_bic <- lm(formula(step_model_bic), data = diag_sample)
manual_model_diag <- lm(formula(manual_model), data = diag_sample)
pca_model_diag <- lm(formula(pca_model_full), data = diag_sample)

# Plot diagnostics for BIC model
print("Saving BIC model diagnostics...")
png(filename = "plots/diagnostics_bic_model.png", width = 1000, height = 1000, res = 120)
par(mfrow = c(2, 2))
plot(step_model_diag_bic, which = 1:4)
par(mfrow = c(1, 1))
dev.off()

# Plot diagnostics for MANUAL model
print("Saving Manual model diagnostics...")
png(filename = "plots/diagnostics_manual_model.png", width = 1000, height = 1000, res = 120)
par(mfrow = c(2, 2))
plot(manual_model_diag, which = 1:4)
par(mfrow = c(1, 1))
dev.off()

# NOT WORKINHG FOR PCA MODEL FOR SOME REASON
# Plot diagnostics for PCA model
print("Saving PCA model diagnostics...")
png(filename = "plots/diagnostics_pca_model.png", width = 1000, height = 1000, res = 120)
par(mfrow = c(2, 2))
plot(pca_model_diag, which = 1:4)
par(mfrow = c(1, 1))
dev.off()

# --- 4. COMPARING THE FINAL MODELS ---
print("--- 4. Final Model Comparison ---")

# Extract Summaries from models fit on FULL data
summary_bic <- summary(step_model_bic)
summary_manual <- summary(manual_model)
summary_pca_full <- summary(pca_model_full)

# Statistics Table
model_stats_final <- data.frame(
  Model = c("Stepwise (BIC)", "Manual (No Fare)", "PCA (Full)"),
  Adj_R_Squared = c(
    summary_bic$adj.r.squared,
    summary_manual$adj.r.squared,
    summary_pca_full$adj.r.squared
  ),
  RSE = c(
    summary_bic$sigma,
    summary_manual$sigma,
    summary_pca_full$sigma
  ),
  Stability_VIF_OK = c("No", "Yes", "Yes"),
  Interpretability = c("Flawed / None", "High", "Very Low (Black Box)")
)

# Print a clean table
print(model_stats_final, digits = 4)
