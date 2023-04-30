# Create edx and final_holdout_test sets 
# Note: this process could take a couple of minutes
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

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")

ratings <- ratings %>%
        mutate(userId = as.integer(userId),
               movieId = as.integer(movieId),
               rating = as.numeric(rating),
               timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
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
edx <- rbind(edx, removed)

rm(dl,ratings_file, movies_file)

# Analysing the dataset.
head(edx,5)
summary(edx)

# Confirming if variables have usable data
sum(is.na.data.frame(edx))

# Review the number of distinct movies, users and genres.
edx %>% summarise(
        uniq_movies = n_distinct(movieId),
        uniq_users = n_distinct(userId),
        uniq_genres = n_distinct(genres))

# Total count of ratings from highest to lowest
edx %>% group_by(rating) %>%
        summarise(Ratings_Count = n()) %>%
        arrange(desc(Ratings_Count))

# Histogram of the spread of Ratings
edx %>% ggplot(aes(rating)) +
        geom_histogram(binwidth = 0.5, color = "black") +
        geom_vline(xintercept = mean(edx$rating), lty = 2) +
        xlab("Rating") +
        ylab("Count") +
        ggtitle("Ratings Histogram") +
        theme(plot.title = element_text(hjust = 0.5))

# View top 5 users by number of ratings
edx %>% group_by(userId) %>%
        summarise(User_Counts = n()) %>%
        arrange(desc(User_Counts)) %>%
        head(5)

# View bottom 5 users by number of ratings
edx %>% group_by(userId) %>%
        summarise(User_Counts = n()) %>%
        arrange(desc(User_Counts)) %>%
        tail(5)

# Histogram of total counts of Ratings by Users 
edx %>% count(userId) %>%
        ggplot(aes(n)) +
        geom_histogram(color = "black", bins=30) +
        geom_vline(xintercept = (nrow(edx)/n_distinct(edx$userId)), lty = 2) +
        scale_x_log10() +
        xlab("# Ratings") +
        ylab("# Users") +
        ggtitle("Counts by Users") +
        theme(plot.title = element_text(hjust = 0.5))

# View top 5 movies by number of ratings
edx %>% group_by(title) %>%
        summarise(Movies_Count = n()) %>%
        arrange(desc(Movies_Count)) %>%
        head(5)

# View bottom 5 movies by number of ratings
edx %>% group_by(title) %>% 
        summarise(Movies_Count = n()) %>%
        arrange(desc(Movies_Count)) %>%
        tail(5)

# Histogram of total counts of Ratings by Movies 
edx %>% count(movieId) %>%
        ggplot(aes(n)) + 
        geom_histogram(color = "black", bins = 30) + 
        scale_x_log10() + 
        xlab("# Count") +
        ylab("# Movies") +
        ggtitle("Count by Movies") +
        theme(plot.title = element_text(hjust = 0.5))     

# Viewing the most popular genres.
# Note: this process could take a couple of minutes
edx_genres <- edx %>% 
        separate_rows(genres, sep = "\\|") %>%
        group_by(genres) %>% 
        summarise(Count = n()) %>% 
        arrange(desc(Count)) 
edx_genres

# Count of Ratings by Genres 
edx_genres %>% 
        ggplot(aes(x = Count, y = reorder(genres,Count))) + 
        geom_col() +
        ylab("Genres") + 
        ggtitle("Count by Genres") + theme(plot.title = element_text(hjust = 0.5)) 

# Creating test and traing sets from the edx set
set.seed(1, sample.kind = "Rounding") # if using R 3.6 or later

test_index <-createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
edx_train <- edx[-test_index,]
edx_temp <- edx[test_index,]

# We need to confirm userId and movieId are in both the train and test sets
edx_test <- edx_temp %>%
        semi_join(edx_train, by = "movieId") %>%
        semi_join(edx_train, by = "userId")

# Add the Rows removed from the edx_test back into edx_train
removed <- anti_join(edx_temp, edx_test)
edx_train <- rbind(edx_train, removed)

rm(edx_temp,removed)

#RMSE function
RMSE <- function(predicted_ratings, true_ratings){
        sqrt(mean((predicted_ratings - true_ratings)^2))
}

# Simple Prediction based on Mean Rating
mu <- mean(edx_train$rating)
mu

# Calculating RMSE using the mean 
rmse_naive <- RMSE(edx_test$rating, mu)
rmse_naive

# Save Results to a Table
rmse_results = tibble(Method = "Naive Analysis by Mean", RMSE = rmse_naive)
rmse_results %>% knitr::kable()

# Model taking into account the movie effects, b_i
movie_avgs <- edx_train %>%
  group_by(movieId) %>%
  summarise(b_i = mean(rating - mu))

predicted_ratings <- mu + edx_test %>%
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

# Calculating RMSE using movie effects 
rmse_model_movie_effects <- RMSE(predicted_ratings, edx_test$rating)
rmse_model_movie_effects

# Adding RMSE results to the Table
rmse_results <- bind_rows(rmse_results, 
                          tibble(Method="Movie Effects Model",
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

# Calculating RMSE using movie and user effects 
rmse_model_user_effects <- RMSE(predicted_ratings, edx_test$rating)
rmse_model_user_effects

# Adding RMSE results to the Table
rmse_results <- bind_rows(rmse_results, 
                          tibble(Method="Movie and User Effects Model",
                                 RMSE = rmse_model_user_effects))
rmse_results %>% knitr::kable()

# Predict via regularisation on the movie and user effect model
# Calculating RMSE using multiple values of lambda.
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
tibble(lambdas,rmses) %>% ggplot(aes(lambdas,rmses)) + geom_point()

# Choose the optimal value of lambda
lambda <- lambdas[which.min(rmses)]
lambda

# Adding RMSE results to the Table
rmse_results <- bind_rows(rmse_results, 
                          tibble(Method="Regularised Movie and User Effects Model",
                                     RMSE = rmse_regularisation))
rmse_results %>% knitr::kable()

# Validating the preferred model using the final_holdout_set.
# Prediction based on Mean Rating on the final_holdout_test set
final_mu <- mean(edx$rating)
final_mu

final_rmse_naive <- RMSE(final_holdout_test$rating, final_mu)
final_rmse_naive

# Save Results in a Table
final_rmse_results = tibble(Method = "Naive Analysis by Mean", RMSE = final_rmse_naive)
final_rmse_results %>% knitr::kable()

# Predict based on Regularisation Model 
min_lambda <- lambda # Using the minimum lambda value from earlier 

b_i <- edx %>%
    group_by(movieId) %>%
    summarise(b_i = sum(rating - final_mu)/(n() + min_lambda))

b_u <- edx %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - b_i - final_mu)/(n() + min_lambda))

predicted_ratings <- final_holdout_test %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = final_mu + b_i + b_u) %>%
    pull(pred)
  
final_rmse_model <- RMSE(final_holdout_test$rating,predicted_ratings)
final_rmse_model

# Save Results to the Table
final_rmse_results <- bind_rows(final_rmse_results, 
                          tibble(Method="Regularised Movie and User Effects Model",
                                     RMSE = final_rmse_model))
final_rmse_results %>% knitr::kable()
