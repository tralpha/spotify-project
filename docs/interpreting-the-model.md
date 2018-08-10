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

# Interpreting The Model

**Random Forest Classifier**

![fig1](images/Feature_Importance.png)

We can note a few things about the features our model found most relevant for making split decsions.  As we suspected, track duration is the most relevant feature by a wide margin.  Most other features are categorical where a large set of classes are possible.  For example, artists that the model found to be good splits were 'Drake', 'Tegan and Sara', 'Deftones', 'Pearl Jam', 'Rihanna', 'Luke Bryan' and 'Kanye West'.  In our earlier Data Description, 'Drake', 'Rihanna', 'Kanye West' and 'Luke Bryan' were in fact found to be in the top 30 artists represented in the MPD, so it's reasonable that a tree would often encounter them and find the information relevant.  It was interesting to see, though, that artist popularity was not the only determining factor in artists making it into the most important features.  Other very popular artists like Eminem, Beyonce and Coldplay did not factor into the top 50 features, whereas less popular artists like Deftones did.  Perhaps people who listen to Deftones tend to listen to only a few songs or are highly likely to listen to a small number of other artists like Deftones.  If that was the case, we could see how Deftones could play a large role in a tree purifying its branches and making correct classifications.  

As with artists, some of the most popular playlist names such as 'country', 'christmas', 'rock', and 'disney' also show up in top features and for good reason.  The variety of songs showing up in the approximately 4,000 playlists named 'disney' in the MPD must be far less than the over 2,000,000 unique songs in MPD, thus, a tree is going to go a long way to purifying its branches by splitting on 'disney'.  
