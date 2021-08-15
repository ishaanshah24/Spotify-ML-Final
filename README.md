# Predicting Song Popularity on Spotify (and creating a song recommender!)
Being a huge fan of music, this project allowed me to explore the different features that contribute to a song's popularity and help predict popularity too. I also created a song and artist recommender using the dataset

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
The dataset was already pretty clean, so most work was done to enhance the existing features and add new ones. Firstly duplicate songs were removed. Certain songs appeared multiple times as they were defined to be in multiple genres (i.e. Hip Hop and Pop). Then, I removed all songs with 0 popularity as there were an overwhelming majority of songs with no popularity. And, also they wouldn't help predict popularity. I also converted the duration column from ms to minutes for simplicity's sake.

To make the dataset more useful for machine learning, the categorical variables of artist name, genre, time_signature, key, and mode were updated using label encoding. Finally, an `is_popular` variable was added. The variable was binary, where songs with over 58 popularity (90th percentile) were represented with a 1 and the rest were represented with a 0.

## Exploratory Data Analysis
Some key insights from the datasets and machine learning models


