# ---
# Title: "Analysis of NYC Yellow Taxi Dataset"
# Author: "Oleksandr Severhin"
# Date: "October 2025"
# Description: This script performs a full analysis of the NYC Yellow Taxi dataset,
#              including data cleaning, EDA, modeling, and diagnostics, as outlined
#              in the Big Data Practice 1 report.
# ---

# --- SECTION 1: SETUP AND CONFIGURATION ---

# Use pacman for efficient package management
if (!require("pacman")) install.packages("pacman")
pacman::p_load(arrow, dplyr, lubridate, ggplot2, ggcorrplot, broom, MASS, patchwork)

# Set a professional theme for all ggplot visuals
theme_set(theme_minimal(base_size = 12))

# Create directories for data and plots if they don't exist
if (!dir.exists("data")) dir.create("data")
if (!dir.exists("plots")) dir.create("plots")


# --- SECTION 2: DATA LOADING ---

# Load the dataset using the arrow package for efficiency
cat("Loading the dataset: data/yellow_tripdata_2025-01.parquet\n")
tryCatch({
  taxi_data_raw <- read_parquet("data/yellow_tripdata_2025-01.parquet")
  cat("Dataset loaded successfully.\n")
}, error = function(e) {
  stop("Failed to load the dataset. Make sure 'yellow_tripdata_2025-01.parquet' is in the 'data' directory.")
})


# --- SECTION 3: DATA CLEANING AND FEATURE ENGINEERING ---

cat("Cleaning data and engineering new features...\n")
taxi_data_clean <- taxi_data_raw %>%
  # 1. Filter out outliers and illogical values
  filter(
    total_amount > 2.5 & total_amount < 250,      # Standard fare starts at $2.50
    fare_amount > 0 & fare_amount < 200,
    trip_distance > 0.1 & trip_distance < 50,     # Very short trips are often errors
    passenger_count > 0 & passenger_count < 7,
    RatecodeID == 1,                              # Standard rate
    payment_type %in% c(1, 2)                     # Card or Cash
  ) %>%
  # 2. Engineer new features from existing columns
  mutate(
    # Create a more descriptive factor for payment_type
    payment_method = factor(payment_type, levels = c(1, 2), labels = c("Card", "Cash")),
    # Extract temporal features using lubridate
    pickup_hour = hour(tpep_pickup_datetime),
    day_of_week = wday(tpep_pickup_datetime, label = TRUE, week_start = 1),
    # Calculate trip duration in minutes
    trip_duration_mins = as.numeric(difftime(tpep_dropoff_datetime, tpep_pickup_datetime, units = "mins")),
    # Calculate average speed
    average_speed_mph = trip_distance / (trip_duration_mins / 60)
  ) %>%
  # 3. Filter based on the newly created features to remove more outliers
  filter(
    trip_duration_mins > 1 & trip_duration_mins < 120,
    !is.na(average_speed_mph) & average_speed_mph > 1 & average_speed_mph < 70
  ) %>%
  # 4. Select the final set of columns for the analysis
  dplyr::select(
    total_amount,
    trip_distance,
    passenger_count,
    payment_method,
    pickup_hour,
    day_of_week,
    trip_duration_mins,
    fare_amount # Keep fare_amount for correlation analysis
  )

cat("Data cleaning complete. Final dataset has", nrow(taxi_data_clean), "rows.\n")

# Save the cleaned data to the 'data' folder
write_parquet(taxi_data_clean, "data/cleaned_yellow_tripdata_2025-01.parquet")
cat("Cleaned data saved to 'data/cleaned_yellow_tripdata_2025-01.parquet'\n")


# --- SECTION 4: INVESTIGATING THE DATA (EXPLORATORY DATA ANALYSIS) ---

cat("Performing Exploratory Data Analysis and generating plots...\n")

# 4.1. Distribution of Key Variables
dist_total_amount <- ggplot(taxi_data_clean, aes(x = total_amount)) +
  geom_histogram(aes(y = ..density..), bins = 50, fill = "steelblue", color = "white", alpha = 0.8) +
  geom_density(color = "darkred", size = 1) +
  labs(title = "Distribution of Total Fare Amount", x = "Total Amount ($)", y = "Density")

dist_trip_distance <- ggplot(taxi_data_clean, aes(x = trip_distance)) +
  geom_histogram(aes(y = ..density..), bins = 50, fill = "skyblue", color = "white", alpha = 0.8) +
  geom_density(color = "darkblue", size = 1) +
  labs(title = "Distribution of Trip Distance", x = "Trip Distance (miles)", y = "Density")

# Display plots together
print(dist_total_amount + dist_trip_distance)

# 4.2. Variable Relationships
# Scatter plot for the report (Figure 1)
scatter_dist_fare <- ggplot(sample_n(taxi_data_clean, 5000), aes(x = trip_distance, y = total_amount)) +
  geom_point(alpha = 0.4, color = "blue") +
  geom_smooth(method = "lm", color = "red", se = FALSE) +
  labs(
    title = "Trip Distance vs. Total Fare Amount",
    subtitle = "A strong positive linear relationship is evident",
    x = "Trip Distance (miles)",
    y = "Total Amount ($)"
  )
ggsave("plots/distance_vs_fare.png", plot = scatter_dist_fare, width = 8, height = 6, dpi = 300)
cat("Saved plot: plots/distance_vs_fare.png\n")

# 4.3. Correlation Insights
# Compute correlation matrix for numeric variables only
numeric_vars <- taxi_data_clean %>% dplyr::select(where(is.numeric))
cor_matrix <- cor(numeric_vars)

# Visualize the correlation matrix
corr_plot <- ggcorrplot(cor_matrix,
  method = "square",
  type = "lower",
  lab = TRUE,
  lab_size = 3,
  colors = c("#6D9EC1", "white", "#E46726")
) +
labs(title = "Correlation Matrix of Numeric Variables")
ggsave("plots/correlation_matrix.png", plot = corr_plot, width = 8, height = 6, dpi = 300)
cat("Saved plot: plots/correlation_matrix.png\n")


# --- SECTION 5: DEVELOPING A PREDICTIVE MODEL ---

cat("Building regression models...\n")

# 5.1. Model 1: Simple Linear Regression
model1 <- lm(total_amount ~ trip_distance, data = taxi_data_clean)
cat("\n--- Summary of Model 1: Simple Linear Regression ---\n")
print(summary(model1))

# 5.2. Model 2: Multiple Linear Regression
# Note: R automatically handles the factor 'payment_method'
model2 <- lm(total_amount ~ trip_distance + passenger_count + payment_method + pickup_hour, data = taxi_data_clean)
cat("\n--- Summary of Model 2: Multiple Linear Regression ---\n")
print(summary(model2))


# --- SECTION 6: EVALUATING MODEL ASSUMPTIONS ---

cat("Generating diagnostic plots for Model 2...\n")

# Use broom to augment the model with diagnostic data
model2_augmented <- augment(model2)

# Residuals vs. Fitted Plot
residuals_plot <- ggplot(sample_n(model2_augmented, 10000), aes(x = .fitted, y = .resid)) +
  geom_point(alpha = 0.3, color = "navy") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(
    title = "Residuals vs. Fitted Values (Model 2)",
    x = "Fitted Values (Predicted Fare)",
    y = "Residuals"
  )
ggsave("plots/residuals_vs_fitted.png", plot = residuals_plot, width = 8, height = 6, dpi = 300)
cat("Saved plot: plots/residuals_vs_fitted.png\n")


# Normal Q-Q Plot
qq_plot <- ggplot(sample_n(model2_augmented, 10000), aes(sample = .resid)) +
  stat_qq(alpha = 0.3) +
  stat_qq_line(color = "red", size = 1) +
  labs(
    title = "Normal Q-Q Plot of Residuals (Model 2)",
    x = "Theoretical Quantiles",
    y = "Sample Quantiles"
  )
ggsave("plots/qq_plot.png", plot = qq_plot, width = 8, height = 6, dpi = 300)
cat("Saved plot: plots/qq_plot.png\n")

cat("\n--- Analysis Complete ---\n")

