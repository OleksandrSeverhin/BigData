# ---
# Title: "Practice 6: Regularization (Ridge, Lasso, E-Net)"
# Author: Oleksandr Severhin
# Date: November, 2025
# ---

# 1. Initialization & Library Management

# Ensure pacman is available
if (!require("pacman")) install.packages("pacman")

# Load dependencies
pacman::p_load(
  arrow,      # Parquet file reading
  dplyr,      # Data manipulation
  ggplot2,    # Plotting
  broom,      # Tidy model outputs
  knitr,      # Table formatting
  caret,      # Sampling/Splitting
  glmnet,     # Ridge, Lasso, Elastic Net
  Metrics,    # RMSE, MAE
  purrr,      # Functional programming tools
  tibble,     # Modern data frames
  tidyr       # Data reshaping
)

# Set Locale
Sys.setlocale("LC_TIME", "English")

# Define Paths
INPUT_PATH <- "data/cleaned_yellow_tripdata_2025-01.parquet"
PLOT_DIR   <- "plots"

# Directory Setup
if (!dir.exists(PLOT_DIR)) {
  dir.create(PLOT_DIR)
  message(sprintf("[Setup] Created directory: %s", PLOT_DIR))
}

# 2. Data Loading & Feature Engineering

if (!file.exists(INPUT_PATH)) {
  stop(sprintf("[Error] File not found: %s", INPUT_PATH))
}

# Load and Transform
# We apply log transformations to address skewness (as per previous analysis)
df_raw <- read_parquet(INPUT_PATH)

df_model <- df_raw %>%
  mutate(
    log_total    = log(total_amount),
    log_distance = log(trip_distance),
    log_fare     = log(fare_amount),
    log_tip      = log(tip_amount + 1),  # +1 to handle zeros
    log_tolls    = log(tolls_amount + 1)
  )

message("[Data] Loaded and transformed successfully.")

# 3. Splitting & Matrix Preparation

# 3.1 Split Data
# Note: glmnet automatically handles standardization (standardize=TRUE by default)
set.seed(123) 

train_idx <- createDataPartition(df_model$log_total, p = 0.8, list = FALSE)
train_set <- df_model[train_idx, ]
test_set  <- df_model[-train_idx, ]

# 3.2 Matrix Creation (One-Hot Encoding)
# We define the formula explicitly to ensure consistency across models
f_model <- as.formula(log_total ~ log_distance + log_fare + log_tip + log_tolls + 
                      passenger_count + payment_type + pickup_hour + 
                      day_of_week + VendorID)

# Generate Model Matrices (removes Intercept column -1 as glmnet adds its own)
X_train <- model.matrix(f_model, data = train_set)[, -1]
X_test  <- model.matrix(f_model, data = test_set)[, -1]

Y_train <- train_set$log_total
Y_test  <- test_set$log_total

# 4. Helper Functions (Refactoring for Modularity)

#' Calculate Performance Metrics
#' Returns a tibble with RMSE, MAE, R2, and hyperparams
evaluate_model <- function(pred_vals, actual_vals, model_name, alpha = NA, lambda = NA) {
  tibble(
    Model     = model_name,
    RMSE      = rmse(actual_vals, pred_vals),
    MAE       = mae(actual_vals, pred_vals),
    R_Squared = R2(actual_vals, pred_vals),
    Alpha     = alpha,
    Lambda    = lambda
  )
}

#' Extract Coefficients
#' Returns a clean dataframe of coefficients for comparison
extract_coefs <- function(model_obj, s_val, col_label) {
  coefs <- coef(model_obj, s = s_val)
  data.frame(
    Term = rownames(coefs),
    Value = as.numeric(coefs)
  ) %>% 
    rename(!!col_label := Value)
}

# Define Lambda Grid for Regularization
lambda_seq <- 10^seq(3, -5, length.out = 150)

# 5. Modeling

# --- A. OLS Baseline (Assignment 3 Reference) ---
fit_ols <- lm(f_model, data = train_set)
pred_ols <- predict(fit_ols, newdata = test_set)

metrics_ols <- evaluate_model(pred_ols, Y_test, "OLS (Baseline)")

# --- B. Ridge Regression (Alpha = 0) ---
set.seed(123) 
cv_ridge <- cv.glmnet(X_train, Y_train, alpha = 0, family = "gaussian", 
                      nfolds = 5, lambda = lambda_seq)

# Plotting Ridge
png(file.path(PLOT_DIR, "01_Ridge_CV.png"), width = 800, height = 600)
plot(cv_ridge, main = "Ridge Regression (Alpha=0): Penalty Search")
dev.off()

metrics_ridge <- bind_rows(
  evaluate_model(predict(cv_ridge, newx=X_test, s="lambda.min"), Y_test, 
                 "Ridge (min)", 0, cv_ridge$lambda.min),
  evaluate_model(predict(cv_ridge, newx=X_test, s="lambda.1se"), Y_test, 
                 "Ridge (1se)", 0, cv_ridge$lambda.1se)
)

# --- C. Lasso Regression (Alpha = 1) ---
set.seed(123)
cv_lasso <- cv.glmnet(X_train, Y_train, alpha = 1, family = "gaussian", 
                      nfolds = 5, lambda = lambda_seq)

# Plotting Lasso
png(file.path(PLOT_DIR, "02_Lasso_CV.png"), width = 800, height = 600)
plot(cv_lasso, main = "Lasso Regression (Alpha=1): Penalty Search")
dev.off()

metrics_lasso <- bind_rows(
  evaluate_model(predict(cv_lasso, newx=X_test, s="lambda.min"), Y_test, 
                 "Lasso (min)", 1, cv_lasso$lambda.min),
  evaluate_model(predict(cv_lasso, newx=X_test, s="lambda.1se"), Y_test, 
                 "Lasso (1se)", 1, cv_lasso$lambda.1se)
)

# --- D. Elastic Net (Grid Search) ---
message("Running Elastic Net Grid Search...")
alpha_candidates <- seq(0, 1, by = 0.1)
results_grid <- data.frame()
best_res <- list(rmse = Inf, model = NULL, alpha = NA)

set.seed(123)
for (a in alpha_candidates) {
  # Train CV model
  fit <- cv.glmnet(X_train, Y_train, alpha = a, family = "gaussian", 
                   nfolds = 5, lambda = lambda_seq)
  
  # Track Best Model
  min_rmse <- sqrt(min(fit$cvm))
  if (min_rmse < best_res$rmse) {
    best_res$rmse  <- min_rmse
    best_res$model <- fit
    best_res$alpha <- a
  }
  
  # Store for plotting
  results_grid <- rbind(results_grid, data.frame(alpha = a, rmse = min_rmse))
}

# Plotting Elastic Net
p_enet <- ggplot(results_grid, aes(x = alpha, y = rmse)) +
  geom_line(color = "#2c3e50", linewidth = 1) +
  geom_point(size = 3, color = "#e74c3c") +
  theme_minimal() +
  labs(title = "Elastic Net: Alpha Optimization",
       subtitle = paste("Optimal Alpha found:", best_res$alpha),
       x = "Alpha Mixing Parameter", y = "CV-RMSE")

ggsave(file.path(PLOT_DIR, "03_ElasticNet_Grid.png"), plot = p_enet, width = 8, height = 6)

metrics_enet <- bind_rows(
  evaluate_model(predict(best_res$model, newx=X_test, s="lambda.min"), Y_test, 
                 "Elastic Net (min)", best_res$alpha, best_res$model$lambda.min),
  evaluate_model(predict(best_res$model, newx=X_test, s="lambda.1se"), Y_test, 
                 "Elastic Net (1se)", best_res$alpha, best_res$model$lambda.1se)
)

# 6. Final Comparison & Interpretation

# 6.1 Performance Table
all_metrics <- bind_rows(metrics_ols, metrics_ridge, metrics_lasso, metrics_enet) %>%
  arrange(RMSE)

print("--- Predictive Performance Comparison ---")
kable(all_metrics, digits = 4, caption = "Model Performance on Test Set") %>% print()

# 6.2 Coefficient Analysis (Interpretation)
coef_table <- list(
    broom::tidy(fit_ols) %>% select(Term = term, OLS = estimate),
    extract_coefs(cv_ridge, "lambda.min", "Ridge_Min"),
    extract_coefs(cv_lasso, "lambda.1se", "Lasso_1se"), # Focus on parsimony
    extract_coefs(best_res$model, "lambda.1se", "ENet_1se")
  ) %>%
  reduce(full_join, by = "Term") %>%
  mutate(across(where(is.numeric), ~ replace_na(., 0))) %>% # Fill dropped vars with 0
  filter(Term != "(Intercept)") %>%
  mutate(across(where(is.numeric), ~ round(., 5)))

print("--- Variable Selection & Shrinkage Analysis ---")
kable(coef_table, caption = "Coefficient Shrinkage: OLS vs Regularized Models") %>% print()
