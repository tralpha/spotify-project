## Table of Contents
[Project Statement and Goals](https://john-daciuk.github.io/spotify/project-statement-and-goals.html) <br>
[Motivation and Background](https://john-daciuk.github.io/spotify/motivation-and-background.html) <br>
[Data Description](https://john-daciuk.github.io/spotify/data-description.html) <br>
[EDA](https://john-daciuk.github.io/spotify/eda.html) <br>
[Data Cleaning](https://john-daciuk.github.io/spotify/data-cleaning.html) <br>
[Metrics](https://john-daciuk.github.io/spotify/metrics.html) <br>
[Data Cleaning](https://john-daciuk.github.io/spotify/data-cleaning.html) <br>
[Model Training](https://john-daciuk.github.io/spotify/model-training.html) <br>
[Interpreting the Model](https://john-daciuk.github.io/spotify/interpreting-the-model.html) <br>
[Model Testing and Results](https://john-daciuk.github.io/spotify/model-testing-and-results.html) <br>
[Literature Review](https://john-daciuk.github.io/spotify/literature-review.html) <br>

# Exploratory Data Analysis:

The data that we have is a trove of information.  The MPD is over 5 gigs of json files of (mostly) random playlists that Spotify has grabbed from unnamed users.  One of the principle challenges of this project will be training on such a quantity of categorical data with limited computing resources.  We intend to take advantage of ‘sklearn’ to train models efficiently. Should ‘sklearn’ give us not so satisfactory metric performance, we’ll try our hands on a Deep Learning library like ‘keras’.

From the MPD, both the playlist length and playlist follower count histograms are approximately power law distributions.  There are a handful of playlists with over 50,000 followers, but the large majority of playlists have only a few followers.  The few playlists with many followers may be disproportionately useful to our recommender system.  The playlists in the data set were selected by Spotify to have between 5 and 250 tracks.  Many playlists naturally have about 25-50 tracks with 66 being the average track count.  

Only 1.9% of the playlists have descriptions, but that still leaves almost 19,000 playlists with descriptions that will likely be helpful.  On average, each song appears in about 30 playlists, and, across all playlists, each unique artists has about 8 unique songs.

The most popular playlist titles are country, chill and rap; the most popular artist, by far, is Drake.  Our preliminary EDA has produced the plots and chart heads below.  We have produced pandas dataframes with the 250 most popular playlist titles, artists and song titles so that any algorithm we employ will have a convenient way to access various measurements of popularity.

We have found redundancy in the playlist dataset and song dataset since each song/ track in the playlist dataset also has meta-data.  We’ve decided that the ‘tags’ in the Million Song Dataset may be useful; however, we plan to first create a sklearn logistic regression model using only the Million Playlist Dataset.  We hypothesize that the MPD may be so rich in song relation information that a robust prediction model may be trained on it alone.  Having access to song lyrics may tempt us to play and experiment with how additional song meta-data can improve our model, although we will try to not overcomplicate.
