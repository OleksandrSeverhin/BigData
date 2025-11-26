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

# 1. ENVIRONMENT SETUP & CONFIGURATION
# Dependency Management
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  arrow,      # I/O
  tidyverse,  # Data Manipulation & Plotting
  broom,      # Tidy model summaries
  caret,      # Splitting & Confusion Matrices
  Metrics,    # RMSE/MAE
  pROC,       # ROC/AUC
  pscl        # Pseudo-R2
)

# Configuration
CONFIG <- list(
  input_path = "data/cleaned_yellow_tripdata_2025-01.parquet", # ENSURE THIS PATH IS CORRECT
  plot_dir   = "plots",
  seed       = 123,
  split_p    = 0.8
)

# Create Output Directory
if (!dir.exists(CONFIG$plot_dir)) dir.create(CONFIG$plot_dir)

# Set global ggplot theme
theme_set(theme_bw() + theme(panel.grid.minor = element_blank()))

# 2. DATA INGESTION & TRANSFORMATION
# Check if file exists before trying to read
if (!file.exists(CONFIG$input_path)) {
  stop(paste("File not found at:", normalizePath(CONFIG$input_path, mustWork = FALSE), 
             "\nPlease check your working directory or move the file to the 'data' folder."))
}

# Load Data
raw_data <- read_parquet(CONFIG$input_path)

# Feature Engineering
# We create a specific dataframe for GLM analysis
df_glm <- raw_data %>%
  mutate(
    # Target 1: Binary classification for Tipping (1 = Tipped, 0 = No Tip)
    is_tipper = if_else(tip_amount > 0, 1, 0),
    
    # Target 2: Continuous Cost
    total_cost = total_amount,
    
    # Log-transformed predictors
    ln_dist  = log(trip_distance),
    ln_fare  = log(fare_amount),
    ln_tolls = log(tolls_amount + 1),
    
    # Factor conversion
    across(c(payment_type, day_of_week, VendorID), as.factor)
  )

# Split Data (Stratified Sampling)
set.seed(CONFIG$seed)
idx_train <- createDataPartition(df_glm$total_cost, p = CONFIG$split_p, list = FALSE)
train_set <- df_glm[idx_train, ]
test_set  <- df_glm[-idx_train, ]

# 3. MODEL A: PREDICTING COST (GAMMA GLM vs OLS)
# --- 3.1 Model Definitions ---
f_cost <- as.formula(total_cost ~ ln_dist + ln_fare + ln_tolls + 
                       passenger_count + pickup_hour + day_of_week + VendorID)

# 1. Baseline: Log-Log OLS (Gaussian)
fit_ols <- lm(log(total_cost) ~ ln_dist + ln_fare + ln_tolls + 
                passenger_count + pickup_hour + day_of_week + VendorID, 
              data = train_set) # FIXED: changed train_data to train_set

# 2. GLM: Gamma with Log Link
fit_gamma <- glm(f_cost, family = Gamma(link = "log"), data = train_set) # FIXED: changed train_data to train_set

# --- 3.2 Performance Comparison ---
pred_ols_raw   <- exp(predict(fit_ols, newdata = test_set))
pred_gamma_raw <- predict(fit_gamma, newdata = test_set, type = "response")

# Comparison Table
perf_compare <- tibble(
  Model_Type = c("OLS (Log-Transformed)", "GLM (Gamma-Log)"),
  Distribution = c("Gaussian (Assumed)", "Gamma"),
  Link_Function = c("Identity (on log y)", "Log"),
  RMSE_Test = c(rmse(test_set$total_cost, pred_ols_raw), 
                rmse(test_set$total_cost, pred_gamma_raw)),
  MAE_Test  = c(mae(test_set$total_cost, pred_ols_raw), 
                mae(test_set$total_cost, pred_gamma_raw)),
  AIC_Score = c(AIC(fit_ols), AIC(fit_gamma)) 
)

print("--- Model A: Performance Comparison ---")
print(kable(perf_compare, digits = 3))

# --- 3.3 Visual Diagnostics ---
df_viz_gamma <- tibble(
  Observed = train_set$total_cost, # FIXED: changed train_data to train_set
  Fitted = fitted(fit_gamma),
  Resid_Pearson = residuals(fit_gamma, type = "pearson")
) %>% sample_n(5000)

p_gamma <- ggplot(df_viz_gamma, aes(x = Fitted, y = Resid_Pearson)) +
  geom_point(alpha = 0.2, color = "#2c3e50") +
  geom_hline(yintercept = 0, color = "#e74c3c", linetype = "dashed") +
  labs(title = "Gamma GLM Diagnostics",
       subtitle = "Pearson Residuals vs Fitted Values (Homoscedasticity Check)",
       x = "Predicted Cost ($)", y = "Pearson Residuals")
ggsave(file.path(CONFIG$plot_dir, "01_Gamma_Residuals.png"), p_gamma, width=7, height=5)

# 4. MODEL B: PREDICTING TIPPING (BINOMIAL GLM)
# Subset for Credit Card only
train_cc <- train_set %>% filter(payment_type == "1")
test_cc  <- test_set  %>% filter(payment_type == "1")

# --- 4.1 Model Fitting ---
# Fit Logistic Regression
fit_logit <- glm(is_tipper ~ trip_distance + fare_amount + tolls_amount + 
                   pickup_hour + day_of_week,
                 family = binomial(link = "logit"),
                 data = train_cc)

# --- 4.2 Interpretation of Coefficients ---
# Extract Odds Ratios
coef_summary <- tidy(fit_logit, exponentiate = TRUE, conf.int = TRUE) %>%
  select(term, estimate, p.value, conf.low, conf.high) %>%
  filter(p.value < 0.05) %>%
  arrange(desc(estimate))

print("--- Model B: Interpretation (Significant Odds Ratios) ---")
print(kable(head(coef_summary, 10), digits = 4, 
            caption = "Top Factors Influencing Likelihood to Tip"))

# --- 4.3 Evaluation (ROC & Pseudo-R2) ---
# Predictions
probs_test <- predict(fit_logit, newdata = test_cc, type = "response")
roc_obj <- roc(test_cc$is_tipper, probs_test)

# Pseudo R-Squared
r2_pseudo <- pR2(fit_logit)["McFadden"]

print(paste0("Binomial GLM - AUC Score: ", round(auc(roc_obj), 4)))
print(paste0("Binomial GLM - McFadden R2: ", round(r2_pseudo, 4)))

p_roc <- ggroc(roc_obj, colour = "steelblue", size = 1) +
  geom_abline(slope = 1, intercept = 1, linetype = "dashed", color = "grey") +
  labs(title = paste0("ROC Curve: Tipping Prediction (AUC = ", round(auc(roc_obj), 2), ")"),
       subtitle = "Binomial GLM Performance on Test Data")
ggsave(file.path(CONFIG$plot_dir, "02_Binomial_ROC.png"), p_roc, width=7, height=5)
