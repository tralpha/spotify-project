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

# Exploratory Data Analysis:

The data that we have is a trove of information.  The MPD is over 5 gigs of json files of (mostly) random playlists that Spotify has grabbed from unnamed users.  One of the principle challenges of this project is training on such a quantity of categorical data with limited computing resources.

It took some time and effort to vectorize the textual information in the dataset and to manage the sparse matrices that result.  In initially considerations about how a decision tree classifier might work on just the quantitative data, track length seemed to have some potential in descriminating songs that fit a playlist from those that don't.  We wondered, given a playlist, what the standard deviation of song duration might look like. If we find that this deviation for playlists tends to be very low, it could give us evidence that song duration is a significant feature. In order to better understand what standard deviation should be considered low, we start by noting that the average track duration across our entire subset of playlists is about 3.9 minutes with a standard deviation of about 1.4 minutes.  Below is plotted the distribution of stds for real (human generated) playlists in our dataset, playlists with many followers and playlists that we randomly generated (see the **data cleaning** section of this site):

![fig1](images/playlist_length_devs.png)

We find that when people create playlists (bottom and middle whisker plots) they tend to vary track duration less than a random algorithm would.  Clearly there is much more to a person’s taste in music than just the length of songs; however, we believe that song length is at least a component of what makes a song enjoyable or not.  Pace contributes to the aura of a harmonious playlist and is palpable to the listener. A song’s length is also likely correlated with other relevant factors such as genre, time period and level of energy. A playlist created for studying would probably benefit from longer songs than one intended for powerlifting.

