# Predicting Song Popularity on Spotify (and creating a song and artist recommender!)
Being a huge fan of music, this project allowed me to explore the different features that contribute to a song's popularity and help predict popularity too. I also created a song and artist recommender using the dataset.

## Project Overview
I spend over 2 hours everyday on Spotify listening to music, creating playlists, and finding new artists. Music helps bring happiness and calmness to my life. Hence, I set out to explore what makes a song and what makes a song popular. My goal was to determine the most influential factors on a song's popularity, and then leverage machine learning to determine an algorithm that will best predict a song’s success. I also planned on using machine learning to create a recommendation system to help me discover new songs and artists. The main language for this project was Python. 

The dataset is linked here: https://www.kaggle.com/zaheenhamidani/ultimate-spotify-tracks-db

## Dataset description 
The dataset had a few techinal music-related variables. They are described below:

* acousticness - A confidence measure from 0.0 to 1.0 of whether the track is acoustic.
* instrumentalness - Predicts whether a track contains no vocals.
* liveness - Detects the presence of an audience in the recording.
* time_signature - The time signature is a notational convention to specify how many beats are in each bar.
* key - Describes the pitch of the song
* mode - The type of scale from which its melodic content is derived (Major or Minor)
* valence - A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. 

Definitions taken from : https://developer.spotify.com/documentation/web-api/reference/#endpoint-get-audio-features

## Preprocessing and Feature Engineering
The dataset was already pretty clean, so most work was done to enhance the existing features and add new ones. Firstly, duplicate songs were removed. Certain songs appeared multiple times as they were defined to be in multiple genres (i.e. Hip Hop and Pop). Then, I removed all songs with 0 popularity as there were an overwhelming majority of songs with no popularity. And, also they wouldn't help predict popularity. I also converted the duration column from ms to minutes for simplicity's sake.

To make the dataset more useful for machine learning, the categorical variables of artist name, genre, time_signature, key, and mode were updated using label encoding. Finally, an `is_popular` variable was added. The variable was binary, where songs with over 58 popularity (90th percentile) were represented with a 1 and the rest were represented with a 0.

## Exploratory Data Analysis
I first found the most popular artists per genre and in the entire dataset. I recognized a lot of familiar names. 

I then made some graphs to better understand the data. 
Some key insights from the datasets:

<img src="https://github.com/ishaanshah24/Spotify-ML-Final/blob/main/Images/Popularity%20Distribution.png" width="350" height="250"/>
Most songs aren't very popular, with the majority of songs between 30-60 popularity.
<img src="https://github.com/ishaanshah24/Spotify-ML-Final/blob/main/Images/Valence%20vs%20Populairty.png" width="350" height="250"/>
It seems that valence doesn't have an impact on popularity, indicating sad or happy songs can be popular.
<img src="https://github.com/ishaanshah24/Spotify-ML-Final/blob/main/Images/Duration%20vs%20Popularity.png" width="350" height="250"/>
Shorter soundtracks tend to be more popular. It looks like songs between 0-10 minutes have the highest popularity.
<img src="https://github.com/ishaanshah24/Spotify-ML-Final/blob/main/Images/Correlation%20Matrix.png" width="350" height="250"/>
Popularity is highly negatively correlated with acousticness, instrumentalness, liveness, and speechiness. This shows that more popular songs tend to be less acoustic, contain vocals, aren't performed live and have less spoken words (more musical).

### Interactive Plot
As we have a plethora of continuous variables, I thought it would be interesting to compare them by genre. But this would create a lot of figures. Hence, I created an interactive plot with drop down menus that allows one to compare features across genres.

*The notebook was too big and hence has been added to [google drive](https://drive.google.com/drive/folders/1h0xMlnvhWxpplTC46kYib2wQotFxoCti) if one would like to view it*

Below are some code snippets and examples of what the plot displays:

<img src="https://github.com/ishaanshah24/Spotify-ML-Final/blob/main/Images/Interactive%20Plot%202.png" width="450" height="500"/>
<img src="https://github.com/ishaanshah24/Spotify-ML-Final/blob/main/Images/Interactive%20Plot%203.png" width="400" height="200"/>
<img src="https://github.com/ishaanshah24/Spotify-ML-Final/blob/main/Images/Interactive%20Plot%204.png" width="400" height="200"/>
<img src="https://github.com/ishaanshah24/Spotify-ML-Final/blob/main/Images/Interactive%20Plot%205.png" width="400" height="200"/>

The interactive plot allowed me to see a lot of interesting relationships and check whether my assumptions about different genres were true or false

## Machine Learning Models

### Regression

#### Linear Regression
Linear regression was the easiest place to start for our regression problem. The features would be used to predict popularity. However, the results weren't great. The R-squared value was 0.205 indicating the model didn't learn too well. The RMSE was 14.5 which is also pretty high, given most songs had a popularity <50.

#### Lasso and Ridge Regression
Lasso and Ridge regression are variants of linear regression. But the main difference is that that it adds a penalty value to the simple linear regression model to prevent overfitting. Ridge regression adds squared magnitude of the coefficient as penalty term while Lasso regression adds absolute value of the coefficient as a penalty term. Lasso regression is also useful for feature selection because it is able to reduce coefficients of insignificant variables in the model to 0.

However, both weren't too succesful at predicting popularity. The best lasso model had a R-squared value of 0.205 with a RMSE of 14.9 and the best ridge model had identical results. Nevertheless, it helped point out which variables were useful.

<img src="https://github.com/ishaanshah24/Spotify-ML-Final/blob/main/Images/Lasso.png" width="750" height="450"/>

Variables close to the zero line are bad predictors. The best predictors look to be acousticness, danceability, energy, liveness, loudness, speechiness, and valence.

I believed that the low R-squared was due to the fact that different genres have different features that make a song in it popular. For example, a rap song maybe more popular due to its lyrics while a jazz song is more popular due to it's instrumentalness. So I ran the model for each genre to check if it would improve accuracy. However, it didn't. Maybe this is because popularity is more random than we believe.


#### Gradient Boosting and Support Vector Regression

Gradient Boost Regression is a type of machine learning boosting model. This means that it uses several different models, with each successive model trying to correct the errors of the last. Gradient boosting aims to predict the target outcomes for the next model to minimize error. The target outcomes are based on the gradient of the error with respect to the prediction. Again, this model wasn't too effective with a R-squared value of 0.32 and RMSE of 13.4. I couldn't tune the hyperparameters for this model as the run time was too long. 

Support Vector Regression attempts to find a hyper plane in an n-dimensional space within a decision boundary that has the maximum amount of points on it. This was an unsuccesful attempt as we only achieved a R-sqaured value of 0.201. The fit time complexity of this model is quadratic and hence it ran very slowly for such a large dataset.

#### Random Forest Regression
Random Forest Regression is a type of ensemble model. This means it combines the predictions from multiple machine learning algorithms together to make more accurate predictions than any individual model. Random Forests using a bagging technqiue where several decision trees run parallely on random samples of the data (with replacement). The final output is a mean of the outputs of each tree. This model got us a R-sqaured value of 0.912 on the training data. However, the R-sqaured value for the test set was only 0.37 which indicates the model overfit the training data. The RMSE was 12.87. We could probably fine-tune the parameters for better results, but the model was taking too long to run.

Nevertheless, we can see the feature importance for each variable through the model. 

<img src="https://github.com/ishaanshah24/Spotify-ML-Final/blob/main/Images/RF.png" width="750" height="450"/>

It seems as if mode, tempo, time-signature, and key were unimportant, while acousticness, loudness, duration, and speechiness were the most important variables.

### Classification

Regression was pretty unsuccesful in predicting popularity. So, I created a new variable - `is_popular` and attempted classification on the dataset. However, the first problem was that the new training dataset was heavily imbalanced. Approximately 90% of the dataset fell in one class and the other 10% in the other class. Hence, when I ran logistic regression model on it, I received a pretty high accuracy of 0.901. This was because all the data was classified as only one class and hence it painted a false picture. Hence, I had to tackle the problem of imbalanced classes.

As this was a large dataset, I decided to randomly undersample the data. This means that points were randomly selected from the majority class until there were the same number of points as the minority class. Now the classes are balanced.

#### Logistic Regression
Logisitic Regression is used to classify data into two or more classes with the help of a sigmoid function. On the undersampled dataset, the logistic regression model returned an accuracy of 0.58 (which means 58% of points were correctly classified). However the precision score was only 0.16 which means that only 16% of points that were predicted to be popular, were actually popular.

#### KNeighbours and Support Vector Classification
A KNeighbours Classifier classifies a point to a class by looking at the K closest points to it. The mode of the class of the K closest points is the class of the point. This model returned an accuracy of 0.59 but a pretty low precision of 0.16. 

Support Vector Classification classifies points by finding the best possible hyperplane to divide the points. The best possible hyperplane is decided with the help of support vectors (points of each class closest to the hyperplane) to maximize the distance between the classes. Here we saw an accuracy of 0.56 with a precision score of 0.16. Not any better than the previous models.

#### Random Forest and Decision Tree Classification

The Random Forest classifier works similarly to the regressor version, with the only difference being that the final class is decided by the mode of the results of each tree. The initial run of this model resulted in an accuracy of 0.65 and a precision score of 0.18. As this was one of the better performing model, its parameters were hypertuned and this resulted in an accuracy of 0.64 and a precision score of 0.19, an insignificant improvement.

A Decision Tree Classifier works by recursively splitting a dataset based on a set of features until all records belong to one class. The initial run of this model resulted in an accuracy of 0.61 and a precision score of 0.15. As this was one of the quicker performing model, its parameters were hypertuned and this resulted in an accuracy of 0.64 and a precision score of 0.16, an insignificant improvement.

#### XGB Classification
XGBoost is a decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting to minimize errors in sequential models. This was the best performing model. The initial run gave an accuracy of 0.66 and a precision score of 0.19. After hyperparameter tuning it gave an accuracy of 0.67 and a precison score of 0.19. This model outperformed the Stacking Classifier that was made too as seen in the AUC scores. XGBoost had an AUC score of 0.78 while the Stacking Classifier had an AUC score of 0.70. This shows that XGBoost is a decent predictor for popularity, but clearly not the best due to its low precision.


### Unsupervised Learning
As a fun final task, I made a song and artist recommender with the given dataset. I used a KNearest Neighbours algorithm to find the songs/artists most similar to the input based on the features in the dataset. The user could choose how many recommendations they want. The biggest roadblock here was to find the exact song name as in the dataset. To avoid this issue, in the future I could connect it to the Spotify API which has a unique track ID for every song. This would allow the user to choose any song (even not in the dataset).

### Conclusion,Learnings & Next Steps
This task was fun and challenging. It looks like a song's popularity depends on more factors than the song's musical features or a song's popularity is more random than we would know. This maybe why we couldn't build a good model. However, this is a question to answer later on. 

Through this journey I got the chance to build on my Python skills and get more comfortable with the Pandas, Plotly, Matplotlib and Sci-Kit Learn libraries. I also better understood the math behind these models and why they work in certain ways. I learnt which models are computationally expensive and which aren't, giving me a clearer idea of which models to use for a dataset in the future. Moreover, I learnt to deal with imbalanced datasets too. 

Some next steps would be to deploy the song and artist recommender so that more people can use it and improve their playlists! I would also try and re-run my models using fewer genres by eliminating genres that are not musical such as Comedy.

If you have read so far, hope you enjoyed and learned something! 🎶👏
