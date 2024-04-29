library(h2o)

# Initialize H2O instance
h2o.init(nthreads = 1)

# Data path
data_path <- "https://github.com/h2oai/h2o-tutorials/raw/master/h2o-world-2017/automl/data/powerplant_output.csv"

# Import data into H2O
df <- h2o.importFile(data_path)

# Split data into training and validation sets
df_splits <- h2o.splitFrame(data = df, ratios = 0.8, seed = 1)
train <- df_splits[[1]]
valid <- df_splits[[2]]
y <- "HourlyEnergyOutputMW"

# Time the model training
start_time <- Sys.time()

powerplant_xgb <- h2o.xgboost(y = y,
                              training_frame = train,
                              validation_frame = valid,
                              ntrees = 2000,
                              seed = 1234)

end_time <- Sys.time()

# Calculate training duration
training_duration <- end_time - start_time
print(paste("Training time: ", training_duration))

# Evaluate model performance
perf <- h2o.performance(powerplant_xgb)
print(perf)

# Generate and print predictions on the validation set
pred <- h2o.predict(powerplant_xgb, newdata = valid)
print(pred)
