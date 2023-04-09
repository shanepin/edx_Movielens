# Create edx and final_holdout_test sets 

# This process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE), stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")

ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE), stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")

movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx,removed)

# Creating separate training and test sets to be used for training, developing and selecting our algorithms.
set.seed(1, sample.kind = "Rounding") # if using R 3.6 or later
test_index <-createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
edx_train <- edx[-test_index,]
edx_temp <- edx[test_index,]

# Confirming userId and movieId are in both the train and test sets
edx_test <- edx_temp %>%
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")

# Add the Rows removed from the edx_test back into edx_train
removed <- anti_join(edx_temp, edx_test)
edx_train <- rbind(edx_train, removed)
```

# Define the RMSE function to be for calculating RMSE's of various models
RMSE <- function(predicted_ratings, true_ratings){
  sqrt(mean((predicted_ratings - true_ratings)^2))
}

#Summarise edx_train
summary(edx_train)

# Movies, Users and Genres in Database
edx_train %>% summarise(
  uniq_movies = n_distinct(movieId),
  uniq_users = n_distinct(userId),
  uniq_genres = n_distinct(genres))

# Ratings Mean 
rating_mean <- mean(edx_train$rating)

# Ratings Histogram
edx_train %>%
  ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.5, color = "black") +
  xlab("Rating") +
  ylab("Count") +
  ggtitle("Ratings Histogram") +
  theme(plot.title = element_text(hjust = 0.5))

# Ratings Users - Number of Ratings
edx_train %>% 
  count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(color = "black", bins=30) +
  scale_x_log10() +
  xlab("# Ratings") +
  ylab("# Users") +
  ggtitle("Users Number of Ratings") +
  theme(plot.title = element_text(hjust = 0.5))

# Ratings Users - Mean by Number with Curve Fitted
edx_train %>%
  group_by(userId) %>%
  summarise(mu_user = mean(rating), number = n()) %>%
  ggplot(aes(x = mu_user, y = number)) +
  geom_bin2d( ) +
  scale_fill_gradientn(colors = grey.colors(10)) +
  labs(fill="User Count") +
  scale_y_log10() +
  geom_smooth(method = lm) +
  ggtitle("Users Average Ratings per Number of Rated Movies") +
  xlab("Average Rating") +
  ylab("# Movies Rated") +
  theme(plot.title = element_text(hjust = 0.5))

# Ratings Movies - Number of Ratings
edx_train %>% 
  count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(color = "black", bins=30) +
  scale_x_log10() +
  xlab("# Ratings") +
  ylab("# Movies") +
  ggtitle("Movie Number of Ratings") +
  theme(plot.title = element_text(hjust = 0.5))

# Simple Prediction based on Mean Rating
mu <- mean(edx_train$rating)
rmse_naive <- RMSE(edx_test$rating, mu)

# Save Results in Data Frame
rmse_results = data_frame(method = "Naive Analysis by Mean", RMSE = rmse_naive)
rmse_results %>% knitr::kable()

# Simple model taking into account the movie effects, b_i
movie_avgs <- edx_train %>%
  group_by(movieId) %>%
  summarise(b_i = mean(rating - mu))

predicted_ratings <- mu + edx_test %>%
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

rmse_model_movie_effects <- RMSE(predicted_ratings, edx_test$rating)
rmse_model_movie_effects
rmse_results <- bind_rows(rmse_results, 
                          data_frame(method="Movie Effects Model",
                          RMSE = rmse_model_movie_effects))
rmse_results %>% knitr::kable()

# Movie and User Effects Model taking into account the user effects, b_u

user_avgs <- edx_train %>%
  left_join(movie_avgs, by="movieId") %>%
  group_by(userId) %>%
  summarise(b_u = mean(rating - mu - b_i))

predicted_ratings <- edx_test %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

rmse_model_user_effects <- RMSE(predicted_ratings, edx_test$rating)
rmse_model_user_effects
rmse_results <- bind_rows(rmse_results, 
                          data_frame(method="Movie and User Effects Model",
                                     RMSE = rmse_model_user_effects))
rmse_results %>% knitr::kable()

# Predict via regularisation, movie and user effect model
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx_train$rating)
  
  b_i <- edx_train %>%
    group_by(movieId) %>%
    summarise(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx_train %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- edx_test %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, edx_test$rating))
  })

rmse_regularisation <- min(rmses)
rmse_regularisation

# Plot RMSE against Lambdas to find optimal lambda
qplot(lambdas, rmses)
lambda <- lambdas[which.min(rmses)]

rmse_results <- bind_rows(rmse_results, 
                          data_frame(method="Regularised Movie and User Effects Model",
                                     RMSE = rmse_regularisation))
# Final table of results
rmse_results %>% knitr::kable()

# Prediction based on Mean Rating on the final_holdout_test set
mu <- mean(edx$rating)
final_rmse_naive <- RMSE(final_holdout_test$rating, mu)

## Save Results in Data Frame
final_rmse_results = data_frame(method = "Naive Analysis by Mean", RMSE = final_rmse_naive)
final_rmse_results %>% knitr::kable()

# Predict via regularisation, movie and user effect model
# We use the minimum lambda value from the earlier models on the final_holdout_set as it was shown to be the best tune.
min_lambda <- lambda

mu <- mean(edx$rating)

b_i <- edx %>%
    group_by(movieId) %>%
    summarise(b_i = sum(rating - mu)/(n() + min_lambda))
b_u <- edx %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - b_i - mu)/(n() + min_lambda))

predicted_ratings <- final_holdout_test %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
final_rmse_model <- RMSE(final_holdout_test$rating,predicted_ratings)
final_rmse_model

## Save Results in Data Frame
final_rmse_results <- bind_rows(final_rmse_results, 
                          data_frame(method="Regularised Movie and User Effects Model",
                                     RMSE = final_rmse_model))
final_rmse_results %>% knitr::kable()
