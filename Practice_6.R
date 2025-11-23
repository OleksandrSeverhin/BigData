# ---
# Title: "Practice 6: Regularization (Ridge, Lasso, E-Net)"
# Author: Oleksandr Severhin
# Date: November, 2025
#
# Desc: This script applies Ridge, Lasso, and Elastic Net regularization
#       to the NYC taxi dataset. It includes data splitting,
#       predictor standardization, and cross-validation to find the
#       optimal penalty. It compares performance against a baseline OLS model.
# ---

## 0. SETUP
if (!require("pacman")) install.packages("pacman")

pacman::p_load(
  arrow,       # Reading parquet
  dplyr,       # Data manipulation
  ggplot2,     # Plotting
  broom,       # Tidy output
  scales,      # Formatting
  rsample,     # Data splitting
  recipes,     # Preprocessing (standardizing)
  glmnet,      # Ridge, Lasso, E-Net
  yardstick,   # Metrics
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
  
  # Filter valid data
  taxi_data_clean <- taxi_data %>%
    filter(fare_amount > 0, trip_distance > 0)
  
  # Sample 100k rows for efficient Cross-Validation execution
  taxi_sample <- taxi_data_clean %>%
    sample_n(min(nrow(.), 100000))
  
  cat(sprintf("Modeling with %d rows.\n", nrow(taxi_sample)))
  
} else {
  stop("Dataset not found.")
}

## 2. PREPROCESSING: SPLIT & STANDARDIZE
model_formula <- fare_amount ~ trip_distance + passenger_count

# Split 80/20
split <- initial_split(taxi_sample, prop = 0.8, strata = fare_amount)
train_data <- training(split)
test_data  <- testing(split)

# Define recipe: Center and Scale all predictors
preproc_recipe <- recipe(model_formula, data = train_data) %>%
  step_center(all_predictors()) %>%
  step_scale(all_predictors())

# Train the recipe on training data
preproc_trained <- prep(preproc_recipe, training = train_data)

# Apply processing (bake) to train and test sets
train_processed <- bake(preproc_trained, new_data = train_data)
test_processed  <- bake(preproc_trained, new_data = test_data)

# Prepare matrices for glmnet
x_train <- as.matrix(train_processed %>% select(-fare_amount))
y_train <- train_processed$fare_amount
x_test  <- as.matrix(test_processed %>% select(-fare_amount))
y_test  <- test_processed$fare_amount

## 3. MODEL FITTING (Cross-Validation)

# A. Baseline OLS
ols_model <- lm(y_train ~ x_train)

# B. Ridge (Alpha = 0)
cv_ridge <- cv.glmnet(x_train, y_train, alpha = 0)
lambda_ridge <- cv_ridge$lambda.min

# C. Lasso (Alpha = 1)
cv_lasso <- cv.glmnet(x_train, y_train, alpha = 1)
lambda_lasso <- cv_lasso$lambda.min

# D. Elastic Net (Alpha = 0.5)
cv_enet <- cv.glmnet(x_train, y_train, alpha = 0.5)
lambda_enet <- cv_enet$lambda.min

## 4. MODEL COMPARISON (Test Set)

# Collect predictions
results <- data.frame(truth = y_test)
results$pred_ols   <- predict(ols_model, newdata = data.frame(x_train = x_test))
results$pred_ridge <- predict(cv_ridge, newx = x_test, s = lambda_ridge)
results$pred_lasso <- predict(cv_lasso, newx = x_test, s = lambda_lasso)
results$pred_enet  <- predict(cv_enet, newx = x_test, s = lambda_enet)

# Calculate metrics
metrics_list <- metric_set(rmse, mae, rsq)

perf_ols   <- metrics_list(results, truth = truth, estimate = pred_ols) %>% mutate(Model = "OLS")
perf_ridge <- metrics_list(results, truth = truth, estimate = pred_ridge) %>% mutate(Model = "Ridge")
perf_lasso <- metrics_list(results, truth = truth, estimate = pred_lasso) %>% mutate(Model = "Lasso")
perf_enet  <- metrics_list(results, truth = truth, estimate = pred_enet) %>% mutate(Model = "Elastic Net")

performance_report <- bind_rows(perf_ols, perf_ridge, perf_lasso, perf_enet) %>%
  tidyr::pivot_wider(names_from = .metric, values_from = .estimate)

print(performance_report)

## 5. PLOTTING AND INTERPRETATION

# Save CV Plots
png("plots/03_ridge_cv.png"); plot(cv_ridge); dev.off()
png("plots/04_lasso_cv.png"); plot(cv_lasso); dev.off()

# Performance Plot
rmse_plot <- ggplot(performance_report, aes(x = Model, y = rmse, fill = Model)) +
  geom_col(show.legend = FALSE) +
  geom_text(aes(label = round(rmse, 4)), vjust = -0.5) +
  labs(title = "RMSE Comparison (Test Set)", y = "RMSE") +
  theme_minimal()

ggsave("plots/05_rmse_comparison.png", rmse_plot, width = 8, height = 5)

# Extract Coefficients at optimal Lambda
coef_ols_vals <- coef(ols_model)
coef_ridge_vals <- coef(cv_ridge, s = lambda_ridge)
coef_lasso_vals <- coef(cv_lasso, s = lambda_lasso)
coef_enet_vals  <- coef(cv_enet, s = lambda_enet)

# Organize coefficients into a table
coef_df <- data.frame(
  Term = rownames(coef_ridge_vals),
  OLS = as.numeric(coef_ols_vals[match(rownames(coef_ridge_vals), names(coef_ols_vals))]),
  Ridge = as.numeric(coef_ridge_vals),
  Lasso = as.numeric(coef_lasso_vals),
  ElasticNet = as.numeric(coef_enet_vals)
)

print(coef_df)

# Visualize Coefficients
coef_plot_long <- coef_df %>%
  tidyr::pivot_longer(cols = -Term, names_to = "Model", values_to = "Coefficient") %>%
  filter(Term != "(Intercept)")

p_coef <- ggplot(coef_plot_long, aes(x = Term, y = Coefficient, fill = Model)) +
  geom_col(position = "dodge") +
  labs(title = "Standardized Coefficient Shrinkage", x = "Predictor") +
  theme_minimal()

ggsave("plots/06_coefficient_shrinkage.png", p_coef, width = 8, height = 5)
