# Sprint Project 06: Sentiment Analysis with NLP
> Sentiment Analysis on Movies Reviews

## Install

You can use `Docker` to easily install all the needed packages and libraries:

```bash
$ docker build -t nlp_project_jc --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) .
```

### Run Docker

```bash
$ docker run --rm --net host -it \
    -v $(pwd):/home/app/src \
    nlp_project_jc \
    bash

$ docker run --rm --net host -it -v "$(pwd)":/home/app/src nlp_project_jc bash
```

## Run Project

It doesn't matter if you are inside or outside a Docker container, in order to execute the project you need to launch a Jupyter notebook server running:

```bash
$ jupyter notebook
```

Then, inside the file `Sentiment_Analysis_NLP.ipynb`, you can see the project statement, description and also which parts of the code you must complete in order to solve it.

## Tests

We've added some basic tests to `Sentiment_Analysis_NLP.ipynb` that you must be able to run without errors in order to approve the project. If you encounter some issues in the path, make sure to be following these requirements in your code:

- Every time you need to run a tokenizer on your sentences, use `nltk.tokenize.toktok.ToktokTokenizer`.
- When removing stopwords, always use `nltk.corpus.stopwords.words('english')`.
- For Stemming, use `nltk.porter.PorterStemmer`.
- For Lematizer, use `Spacy` pre-trained model `en_core_web_sm`.

You can use others methods if you want to do extra experimentation but do it outside the code used to run the tests. Otherwise, they may fail for some specific cases.
