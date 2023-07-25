import sys
import pandas as pd
import numpy as np
import re
import nltk
import os
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

import pprint

from Tagalog_Words import tagalog_stop_words

# pd.options.display.max_rows = 300
tqdm.pandas()
vectorizer = TfidfVectorizer()
wordnet_lemmatizer = WordNetLemmatizer()


# tokenize, remove stop and stemmed words (english p lang)
def stop_stemmed(comment):
    """
    Tokenizes the given comment and removes the stop words. Then, it applies stemming to the remaining tokens and returns the stemmed tokens.

    Args:
        comment (str): The input comment to be processed.

    Returns:
        list: A list of stemmed tokens after removing the stop words.
    """

    # tokenize each comment from past data
    tokens = word_tokenize(comment)

    stop_words = set(stopwords.words("english"))

    stop_words.update(tagalog_stop_words)

    # all stop words (from nltk.corpus & tagalog)

    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

    # stemming the word before applying stop word removal
    stemmer = SnowballStemmer("english")
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

    # # defining the object for Lemmatization
    # baka palitan tong lemmatier

    lemmate_tokens = [wordnet_lemmatizer.lemmatize(word) for word in stemmed_tokens]

    # return array  of string to string
    orig = " ".join(lemmate_tokens)

    return orig


# --------
def preprocess_comment_text(comment):
    replacements = [("performance", ""), ("product quality", ""), ("best feature", "")]

    # to lowercase all words
    comment = comment.lower()

    # remove special chars and digits
    comment = re.sub(r"[^a-zA-Z]", " ", comment, flags=re.UNICODE)

    # remove the word that is frequent and not needed from string
    for old, new in replacements:
        comment = re.sub(old, new, comment, count=1, flags=re.UNICODE)
    # remove extra whitespaces inside the string
    comment = re.sub(r"\s+", " ", comment, flags=re.UNICODE)

    # using python re.sub create a new string removing the first  thank word
    comment = re.sub(r"thank\s+", "", comment, flags=re.UNICODE)

    # removes first and last space
    comment = comment.strip()

    # Perform the Contractions on the reviews.
    comment = re.sub(r"won\’t", "will not", comment)
    comment = re.sub(r"would\’t", "would not", comment)
    comment = re.sub(r"could\’t", "could not", comment)
    comment = re.sub(r"\’d", " would", comment)
    comment = re.sub(r"can\’t", "can not", comment)
    comment = re.sub(r"n\’t", " not", comment)
    comment = re.sub(r"\’re", " are", comment)
    comment = re.sub(r"\’s", " is", comment)
    comment = re.sub(r"\’ll", " will", comment)
    comment = re.sub(r"\’t", " not", comment)
    comment = re.sub(r"\’ve", " have", comment)
    comment = re.sub(r"\’m", " am", comment)

    return comment


# --------


# starting point of the program
def main():
    """
    The main function.
    """
    print("Program Entry Point")

    # load the data
    datasource = pd.read_csv("datasource.csv")

    # drop the columns that we dont need for the analysis
    to_drop_coloums = ["username"]

    # print dataframe shape and columns
    print(f"\nShape:{datasource.shape}\n\nColumn Names:\n{datasource.columns}\n")

    # drop the columns
    datasource.drop(columns=to_drop_coloums, axis=1, inplace=True)

    # number of rows of the datasoruce before droping rows with no values
    print("current data is", len(datasource.index))

    # drop the rows with no values
    datasource.dropna(how="all", inplace=True)

    # number of rows of the datasoruce after droping rows with no values
    print("current data is after droping rows with no values", len(datasource.index))

    # remove duplicate rows
    datasource.drop_duplicates(inplace=True)

    # number of rows of the datasoruce after removing duplicates
    print(
        " number of rows of the datasoruce after removing duplicates",
        len(datasource.index),
    )

    # fill the empty cells with "No review"
    datasource["comment"].fillna("", inplace=True)

    # fill the empty cells with "3"
    datasource["rating"].fillna("3", inplace=True)

    print(datasource.head())

    # lowercase all the comments/remove special chars and digits/remove extra whitespaces
    datasource["comment"] = datasource["comment"].progress_apply(
        lambda x: preprocess_comment_text(x)
    )

    print(datasource.head())

    # drop the rows containing spaces only
    # datasource = datasource[datasource["comment"].str.strip() != ""]

    # number of rows of the datasoruce after dropping all the rows with no values

    print("current data is", len(datasource.index))

    # ---------------------------------------------------------------
    # store filtered comments in stopstemmed.csv
    # preprocessed_data = datasource["comment"].progress_apply(stop_stemmed)

    # store filtered comments in stopstemmed.csv
    datasource["processed_comment"] = datasource["comment"].progress_apply(stop_stemmed)

    # print(datasource["rating"])

    # convert the star_rating column to int
    datasource["rating"] = datasource["rating"].astype(int)

    #  2=Positve,1=Nuetral, 0=Negative
    # datasource["label"] = np.where(datasource["rating"] >= 4, 1, 0)

    datasource["label"] = np.where(
        datasource["rating"] < 2, 0, np.where(datasource["rating"] == 3, 1, 2)
    )

    # print(datasource["label"].value_counts())
    # datasource.head()

    # print(datasource.head())

    # save the current instance of stemstmed
    datasource.to_csv("preprocessed_data.csv", index=False)
    # ---------------------------------------------------------------

    # print(datasource["label"].values)

    plt.hist(datasource["label"].values, bins=3, align="mid")
    plt.xticks(range(3), ["Negative", "Neutral", "Positive"])
    plt.xlabel("Sentiment of Reviews")
    plt.title("Distribution of Sentiment")
    plt.savefig("sentiment.png")
    # plt.show()

    # ---------------------------------------------------------------

    # 'X_train' will contain the features for training.
    # 'y_train' will contain the corresponding target labels for training.
    # 'X_test' will contain the features for testing.
    # 'y_test' will contain the corresponding target labels for testing.

    X_train, X_test, Y_train, Y_test = train_test_split(
        datasource["processed_comment"],
        datasource["label"],
        test_size=0.3,
        random_state=42,
    )

    print(
        "Train: ", X_train.shape, Y_train.shape, "Test: ", (X_test.shape, Y_test.shape)
    )

    # Step 2: Vectorize the text data
    print("TFIDF Vectorizer……")

    tf_x_train = vectorizer.fit_transform(X_train)

    tf_x_test = vectorizer.transform(X_test)

    # # ------------------------------
    # SVM (Support Vector Machine)

    # Step 3: Choose a Machine Learning Model
    clf = LinearSVC(random_state=0)

    # Step 4: Train the Model
    clf.fit(tf_x_train, Y_train)

    # Step 5: Evaluate the Model
    y_test_pred = clf.predict(tf_x_test)
    accuracy = accuracy_score(Y_test, y_test_pred)
    # print(f"Accuracy: {accuracy}")

    report = classification_report(Y_test, y_test_pred, output_dict=True)

    print("---------- START REPORT FROM SVM ----------")
    print(f"Accuracy: {accuracy}")
    pprint.pprint(report)
    print("---------- END REPORT FROM SVM ----------")

    # # ---------------------------------------------
    # # sentence prediction using the SVM

    # this will apply a prediction on each sentence
    datasource["prediction"] = clf.predict(
        vectorizer.transform(pd.Series(datasource["processed_comment"]))
    )

    print(datasource["prediction"].describe())

    datasource.to_csv("finaltest.csv", index=False)
    # # ---------------------------------------------------------------

    # convert to numeric values
    print("------------------------------------------")
    print("Sentiment Analysis")
    print(datasource["prediction"].value_counts())
    print("------------------------------------------")

    # # ------------------------------
    # # Logistic Regression

    lra = LogisticRegression(max_iter=1000, solver="saga")

    # Fit the Training data to the model
    lra.fit(tf_x_train, Y_train)

    # Predicting the test data
    y_test_pred = lra.predict(tf_x_test)

    # Analyzing the Report
    report = classification_report(Y_test, y_test_pred, output_dict=True)
    accuracy = accuracy_score(Y_test, y_test_pred)

    print("---------- START REPORT FROM LOGISTIC REGRESSION ----------")
    print(f"Accuracy: {accuracy}")
    pprint.pprint(report)
    print("---------- END REPORT FROM LOGISTIC REGRESSION ----------")

    # Convert the training and testing sets to pandas DataFrames.
    train_df = pd.DataFrame({"comment": X_train, "label": Y_train})
    test_df = pd.DataFrame({"comment": X_test, "label": Y_test})

    # Save the DataFrames to CSV files.
    train_df.to_csv("train_data.csv", index=False)
    test_df.to_csv("test_data.csv", index=False)

    # a wordcloud for negative comments
    # Polarity == 0 negative
    train_s0 = train_df[train_df.label == 0]
    all_text = " ".join(
        wordnet_lemmatizer.lemmatize(word) for word in train_s0["comment"]
    )
    wordcloud = WordCloud(
        colormap="Reds",
        width=1000,
        height=1000,
        mode="RGBA",
        background_color="white",
    ).generate(all_text)
    plt.title("Negative Comments")
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig("negative.png")
    # plt.show()

    # a wordcloud for neutral comments
    # # Polarity == 1 neutral
    train_s0 = train_df[train_df.label == 1]
    all_text = " ".join(
        wordnet_lemmatizer.lemmatize(word) for word in train_s0["comment"]
    )
    wordcloud = WordCloud(
        colormap="Blues",
        width=1000,
        height=1000,
        mode="RGBA",
        background_color="white",
    ).generate(all_text)
    plt.title("Neutral Comments")
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig("neutral.png")
    # plt.show()

    # a wordcloud for positive comments
    # # Polarity == 2 positive
    train_s0 = train_df[train_df.label == 2]
    all_text = " ".join(
        wordnet_lemmatizer.lemmatize(word) for word in train_s0["comment"]
    )
    wordcloud = WordCloud(
        colormap="Greens",
        width=1000,
        height=1000,
        mode="RGBA",
        background_color="white",
    ).generate(all_text)
    plt.title("Positive Comments")
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig("positive.png")
    # plt.show()

    #

    # plt.hist(datasource.prediction, bins=2, align="mid")
    # plt.xticks(range(2), ["Negative", "Positive"])
    # plt.xlabel("Sentiment of Reviews")
    # plt.title("Distribution of Sentiment")
    # plt.show()

    # print("------------------------------------------")
    # print("Rating Analysis")
    # print(datasource["label"].value_counts())
    # print("------------------------------------------")

    # plt.hist(datasource.label, bins=2, align="mid")
    # plt.xticks(range(2), ["Negative", "Positive"])
    # plt.xlabel("Sentiment of Reviews")
    # plt.title("Distribution of Sentiment")
    # plt.show()

    # generate wordcloud
    generate_wordcloud(dataframe=datasource)

    # calling the lenght plotter
    lenght_plotter(datasource)


def generate_wordcloud(dataframe):
    text = " ".join(dataframe["comment"].tolist())
    # towordcloud = dataframe["comment"][0]

    # Create and generate a word cloud image:
    wc = WordCloud(
        max_words=250,
        width=2300,
        height=1600,
    )
    wordcloud = wc.generate(text)

    # Display the generated image:
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")

    # save the wordcloud
    plt.savefig("wordcloud.jpg", dpi=300)

    # show the wordcloud
    # plt.show()
    plt.close()


def lenght_plotter(dataframe):
    """
    Generate a plot to visualize the length distribution of comments in a given dataframe.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe containing the comments.

    Returns:
    None
    """
    dataframe["rowlenght"] = dataframe["comment"].apply(len)
    plt.hist(dataframe["rowlenght"].values, bins=range(0, 1000, 10), ec="black")
    plt.xlabel("Number of characters")
    plt.ylabel("Number of comments")
    plt.title("Number of characters and number of comments")
    plt.savefig("lenghts.jpg", dpi=300)
    # plt.show()


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    # start = time.time()
    main()
    # end = time.time()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
