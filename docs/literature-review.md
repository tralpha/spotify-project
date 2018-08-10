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

# Literature Review

[Evaluation in Information Retrieval](https://nlp.stanford.edu/IR-book/pdf/08eval.pdf) and [A Review on Evaluation Metrics for Data Classification Evaluations](https://pdfs.semanticscholar.org/6174/3124c2a4b4e550731ac39508c7d18e520979.pdf).
We’re including the above two papers for review because we believe that this task is very similar to an information retrieval task, where given a query (features extract from a Playlist), we retrieve documents (audio songs). Reading through the above book chapter will enable us to understand more profoundly how such a system is evaluated, which will shed light on how our metrics should be computed.
Natural Language Processing in Information Retrieval and Neural Methods for Information Retrieval
These papers will assist us in choosing a suitable model for the problem at hand. Our primary plan is to vectorize the metadata word tokens of the songs, and even other information such as the time the song was produced, and combine these together, then classify these playlist-song pairs, in order to come up with a score of whether the playlist-song pair is a good one (1) or it’s a bad one (0). We intend to use `sklearn` for this task, but in case we use a more complex model, we might use `keras` or `xgboost` libraries. 
