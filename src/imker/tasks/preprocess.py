import imker
import neattext as nt
import pandas as pd
from nltk.stem import PorterStemmer, StemmerI


class TextCleansingTask(imker.BaseTask):  # type: ignore
    def __init__(
        self,
        text_cols: list[str] = ["text"],
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
        self.text_cols = text_cols
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

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        output_df = pd.DataFrame()
        for text_col in self.text_cols:
            df = self.clean(X=X, text_col=text_col)
            output_df = pd.concat([output_df, df], axis=1)

        return output_df

    def clean(self, X: pd.DataFrame, text_col: str) -> pd.DataFrame:
        output_df = pd.DataFrame()
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
            for text in X[text_col]
        ]
        if self.stemmer:
            cleansed_texts = [self.stemming(text, stemmer=self.stemmer) for text in cleansed_texts]
        output_df[f"{text_col}_cleansed"] = cleansed_texts
        return output_df

    @staticmethod
    def stemming(text: str, stemmer: StemmerI = PorterStemmer()) -> str:
        return " ".join([stemmer.stem(word) for word in text.split()])
