### Table of Contents
[Project Statement and Goals](https://tralpha.github.io/spotify-project/project-statement-and-goals.html) <br>
[Motivation and Background](https://tralpha.github.io/spotify-project/motivation-and-background.html) <br>
[Data Description](https://tralpha.github.io/spotify-project/data-description.html) <br>
[EDA](https://tralpha.github.io/spotify-project/eda.html) <br>
[Data Cleaning](https://tralpha.github.io/spotify-project/data-cleaning.html) <br>
[Metrics](https://tralpha.github.io/spotify-project/metrics.html) <br>
[Model Training](https://tralpha.github.io/spotify-project/model-training.html) <br>
[Interpreting the Model](https://tralpha.github.io/spotify-project/interpreting-the-model.html) <br>
[Model Testing and Results](https://tralpha.github.io/spotify-project/model-testing-and-results.html) <br>
[Conclusion and What's Next](https://tralpha.github.io/spotify-project/conclusion.html) <br>
[Literature Review](https://tralpha.github.io/spotify-project/literature-review.html) <br>

### Authors
Ralph Tigoumo, Fredrik Karlsson Peraldi and John Daciuk

# Project Statement:
With music streaming services, many of us have access to more music than we'll ever have time to explore on our own.  We build a system that can recommend new songs that will likely be compelling to a user.  A playlist that the user has already created will be used as ground truth for the user's taste in music.

## Goals:
1. Given a playlist from Spotify's Million Playlist Dataset as input, output a list of tracks that will be reasonable additions for continuing that playlist.

2. Measure an R-Precision score that can give quantitative data indicating how our system compares to the state of the art and confirm that our model has successfully learned trends in the training data that generalize.

3. Interpret our model to understand what features in the MPD are most relevant to predicting songs that contribute to a harmonious playlist.

4. Add functionality to make recommendations for our *own* playlists
