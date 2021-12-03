import re
from nltk.stem import WordNetLemmatizer
from nltk.stem import *


class DataCleaning():
    def __init__(self, data):
        self.data = data

    # Removes Punctuations
    def remove_punctuations(self, text):
        punct_tag=re.compile(r'[^\w\s]')
        data=punct_tag.sub(r'', text)
        return data

    # lower case
    def lower_case(self, text):
        return str(text).lower()

    # Remove any url
    def remove_URL(self, text):
        return re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove any HTML tags
    def remove_html(self, text):
        html = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
        return re.sub(html, "", text)

    # Remove Non-ASCI
    def remove_non_ascii(self, text):
        return re.sub(r'[^\x00-\x7f]',r'', text)

    # Remove Special Characters
    def remove_special_characters(self, text):
        emoji_pattern = re.compile(
            '['
            u'\U0001F600-\U0001F64F'  # emoticons
            u'\U0001F300-\U0001F5FF'  # symbols & pictographs
            u'\U0001F680-\U0001F6FF'  # transport & map symbols
            u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
            u'\U00002702-\U000027B0'
            u'\U000024C2-\U0001F251'
            ']+',
            flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    # Lemmatize the text
    def lemma_traincorpus(self, text):
        lemmatizer=WordNetLemmatizer()
        out_data=""
        for words in text:
            out_data+= lemmatizer.lemmatize(words)
        return out_data

    # Stem text
    def stem_traincorpus(self, text):
        stemmer = PorterStemmer()
        out_data=""
        for words in text:
            out_data+= stemmer.stem(words)
        return out_data

    def clean_data(self):
        text = self.data
        # remove punctuation
        text = text.apply(lambda x: self.remove_punctuations(x))
        # lower case
        text = text.apply(lambda x: self.lower_case(x))
        # Remove url
        text = text.apply(lambda x: self.remove_URL(x))
        # Remove url
        text = text.apply(lambda x: self.remove_html(x))
        # Remove non ascii
        text = text.apply(lambda x: self.remove_non_ascii(x))
        # Remove non ascii
        text = text.apply(lambda x: self.remove_special_characters(x))
        # Lemmatize
        text = text.apply(lambda x: self.lemma_traincorpus(x))
        # Stem
        text = text.apply(lambda x: self.stem_traincorpus(x))

        return text




