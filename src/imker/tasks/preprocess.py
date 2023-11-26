from typing import Any

import cupy as cp
import imker
import neattext as nt
import pandas as pd
from cuml.feature_extraction.text import TfidfVectorizer as TfidfVectorizer_gpu
from nltk.stem import PorterStemmer, StemmerI
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm


class ExtractRawFeaturesTask(imker.BaseTask):  # type: ignore
    def __init__(self, base_columns: list[str]) -> None:
        self.base_columns = base_columns

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.base_columns]


class ExtractTfIdfFeaturesTask(imker.BaseTask):  # type: ignore
    def __init__(self, text_columns: list[str] = ["text"], use_gpu: bool = True, **kwargs: dict[str, Any]) -> None:
        self.kwargs = kwargs
        self.text_columns = text_columns
        self.vectorizers: dict[str, TfidfVectorizer] = {}

        self.kwargs["ngram_range"] = tuple(self.kwargs["ngram_range"])  # type: ignore

        self.use_gpu = use_gpu

    @property
    def vectorizer(self) -> TfidfVectorizer | TfidfVectorizer_gpu:
        return self._vectorizer

    @vectorizer.setter
    def vectorizer(self, val: TfidfVectorizer | TfidfVectorizer_gpu) -> None:
        self._vectorizer = val

    def reset_vectorizer(self) -> None:
        if self.use_gpu:
            self.vectorizer = TfidfVectorizer_gpu(**self.kwargs)
        else:
            self.vectorizer = TfidfVectorizer(**self.kwargs)

    def fit(self, X: pd.DataFrame) -> "ExtractTfIdfFeaturesTask":
        for text_col in self.text_columns:
            self.vectorize(X=X, text_col=text_col)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        output_df = pd.DataFrame()
        for text_col in self.text_columns:
            vectorizer = self.vectorizers[text_col]
            tfidf_features = vectorizer.transform(X[text_col]).toarray()
            if self.use_gpu:
                tfidf_features = cp.asnumpy(tfidf_features)
            tfidf_features = pd.DataFrame(
                tfidf_features,
                columns=[f"{text_col}_tfidf_{i:03}" for i in range(tfidf_features.shape[1])],
            )
            output_df = pd.concat([output_df, tfidf_features], axis=1)

        return output_df.add_prefix("f_")

    def vectorize(self, X: pd.DataFrame, text_col: str) -> None:
        self.reset_vectorizer()
        self.vectorizer.fit(X[text_col])
        self.vectorizers[text_col] = self.vectorizer


class TextCleansingTask(imker.BaseTask):  # type: ignore
    def __init__(
        self,
        puncts: bool = True,
        stopwords: bool = False,
        urls: bool = True,
        emails: bool = True,
        numbers: bool = True,
        emojis: bool = True,
        special_char: bool = True,
        phone_num: bool = True,
        non_ascii: bool = True,
        multiple_whitespaces: bool = True,
        contractions: bool = True,
        currency_symbols: bool = True,
        custom_pattern: str | None = None,
        stemmer: StemmerI = PorterStemmer(),
    ) -> None:
        self.puncts = puncts
        self.stopwords = stopwords
        self.urls = urls
        self.emails = emails
        self.numbers = numbers
        self.emojis = emojis
        self.special_char = special_char
        self.phone_num = phone_num
        self.non_ascii = non_ascii
        self.multiple_whitespaces = multiple_whitespaces
        self.contractions = contractions
        self.currency_symbols = currency_symbols
        self.custom_pattern = custom_pattern
        self.stemmer = stemmer

    def transform(self, X: list[str]) -> pd.DataFrame:
        cleansed_texts = [
            nt.clean_text(
                text=text,
                puncts=self.puncts,
                stopwords=self.stopwords,
                urls=self.urls,
                emails=self.emails,
                numbers=self.numbers,
                emojis=self.emojis,
                special_char=self.special_char,
                phone_num=self.phone_num,
                non_ascii=self.non_ascii,
                multiple_whitespaces=self.multiple_whitespaces,
                contractions=self.contractions,
                currency_symbols=self.currency_symbols,
                custom_pattern=self.custom_pattern,
            )
            for text in tqdm(X, desc="Cleansing")
        ]
        if self.stemmer:
            cleansed_texts = [
                self.stemming(text, stemmer=self.stemmer) for text in tqdm(cleansed_texts, desc="Stemming")
            ]
        return cleansed_texts

    @staticmethod
    def stemming(text: str, stemmer: StemmerI = PorterStemmer()) -> str:
        return " ".join([stemmer.stem(word) for word in text.split()])


class TfIdfVectorizerTask(imker.BaseTask):  # type: ignore
    def __init__(self, **kwargs: dict[str, Any]) -> None:
        self.kwargs = kwargs
        self.kwargs["ngram_range"] = tuple(self.kwargs["ngram_range"])  # type: ignore

    def fit(self, X: list[str]) -> "TfIdfVectorizerTask":
        self.vectorizer = TfidfVectorizer(**self.kwargs)
        self.vectorizer.fit(X)
        return self

    def transform(self, X: list[str]) -> pd.DataFrame:
        return self.vectorizer.transform(X)


class CountVectorizerTask(imker.BaseTask):  # type: ignore
    def __init__(self, **kwargs: dict[str, Any]) -> None:
        self.kwargs = kwargs
        self.kwargs["ngram_range"] = tuple(self.kwargs["ngram_range"])  # type: ignore

    def fit(self, X: list[str]) -> "CountVectorizerTask":
        self.vectorizer = CountVectorizer(**self.kwargs)
        self.vectorizer.fit(X)
        return self

    def transform(self, X: list[str]) -> pd.DataFrame:
        return self.vectorizer.transform(X)
