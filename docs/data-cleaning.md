## Table of Contents
[Project Statement and Goals](https://tralpha.github.io/spotify-project/project-statement-and-goals.html) <br>
[Motivation and Background](https://tralpha.github.io/spotify-project/motivation-and-background.html) <br>
[Data Description](https://tralpha.github.io/spotify-project/data-description.html) <br>
[EDA](https://tralpha.github.io/spotify-project/eda.html) <br>
[Data Cleaning](https://tralpha.github.io/spotify-project/data-cleaning.html) <br>
[Metrics](https://tralpha.github.io/spotify-project/metrics.html) <br>
[Data Cleaning](https://tralpha.github.io/spotify-project/data-cleaning.html) <br>
[Model Training](https://tralpha.github.io/spotify-project/model-training.html) <br>
[Interpreting the Model](https://tralpha.github.io/spotify-project/interpreting-the-model.html) <br>
[Model Testing and Results](https://tralpha.github.io/spotify-project/model-testing-and-results.html) <br>
[Literature Review](https://tralpha.github.io/spotify-project/literature-review.html) <br>

# Data Cleaning:
Vectorization, transfer to sparse matrix, merging of playlist and song data, creating negative samples to train on, creating massive track list to predict on.

We started by creating 3 separate data frames from the MPD: one for global playlist features (one row per unique playlist), one for track features (one row per unique track) and one to map tracks to playlists.  In order to explore the data on the fly, we limit the number of playlists loaded into our notebook to a few thousand.  Finally, we merge the data frames to create one frame, ‘merged’, that has one track per row. The features of each track are the features of the playlist from which it came from together with the features of the track itself. 

As an example, if we loaded n playlists each containing m songs, then our merged dataframe will have n x m rows.
After feature engineering, we create our label variable, which is called `match`. `match` is described below:
Match: All track rows are set ‘match’ = 1, signifying that they belong to their respective playlists.  In order to aid the training of any model, we also appended to ‘merged’ a negative sample dataframe. The negative sample frame is a copy of ‘merged’, but with all the tracks randomly assigned to playlists with ‘match’ column set to 0. Each playlist’s positive sample to negative sample ratio is 1:1.
