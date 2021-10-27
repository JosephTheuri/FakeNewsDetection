import string
from textblob import TextBlob
from nltk.corpus import stopwords

class FeatureExtraction():
    def __init__(self, data, language):
        self.data = data
        self.language = language

    # Text based features
    def text_features(self):
        df = self.data

        """
       
        Feature 1: Word Count in the Essay - total number of words in the complete essay text
        Feature 2: Character Count in the Complete Essay - total number of characters in complete essay text
        Feature 3: Word Density of the Complete Essay - average length of the words used in the essay
        Feature 4: Puncutation Count in the Complete Essay - total number of punctuation marks in the essay
        Feature 5: Upper Case Count in the Complete Essay - total number of upper count words in the essay
        Feature 6: Title Word Count in the Complete Essay - total number of proper case (title) words in the essay
        Feature 7: Stopword Count in the Complete Essay - total number of stopwords in the essay
       
        """

        # Helper functions
        punctuation = string.punctuation
        stop_words = list(set(stopwords.words(self.language)))

        # Extract features from text
        df['char_count'] = df['text'].apply(len)    # Calculate count of characters
        df['word_count'] = df['text'].apply(lambda x: len(x.split()))   # Calculate count of words
        df['word_density'] = df['char_count'] / (df['word_count']+1)    # Calculate word density
        df['punctuation_count'] = df['text'].apply(lambda x: len("".join(_ for _ in x if _ in punctuation)))    # Calculate punctuation count
        df['title_word_count'] = df['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))   # Calculate title case count
        df['upper_case_word_count'] = df['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))  # Calculate upper case count
        df['stopword_count'] = df['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.lower() in stop_words])) # Calculate stop word case count

        return df

    # Grammatic features
    def grammatic_features(self):
        df = self.data

        """
        Part of Speech related features from the text

        
        Feature 8: Article Polarity - polarity score of essay
        Feature 9: Article Subjectivity - subjectivity score of essay
        Feature 10: Noun Count - proportion of nouns in essay
        Feature 11: Verb Count - proportion of verbs in essay
        Feature 12: Adjective Count - proportion of adjectives in essay
        Feature 13: Adverb Count - proportion of adverbs in essay
        Feature 14: Pronoun Count - proportion of pronouns in essay

        """

        # Helper Functions
        ## Specify POS dict
        pos_dic = {
            'noun' : ['NN','NNS','NNP','NNPS'],
            'pron' : ['PRP','PRP$','WP','WP$'],
            'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
            'adj' :  ['JJ','JJR','JJS'],
            'adv' : ['RB','RBR','RBS','WRB']
        }

        ## Check and get the part of speech tag count of a words in a given sentence
        def pos_check(x, flag):
            cnt = 0
            try:
                wiki = TextBlob(x)
                for tup in wiki.tags:
                    ppo = list(tup)[1]
                    if ppo in pos_dic[flag]:
                        cnt += 1
            except:
                pass
            return cnt

        # Extract grammatic featues
        df['noun_count'] = df['text'].apply(lambda x: pos_check(x, 'noun'))
        df['verb_count'] = df['text'].apply(lambda x: pos_check(x, 'verb'))
        df['adj_count'] = df['text'].apply(lambda x: pos_check(x, 'adj'))
        df['adv_count'] = df['text'].apply(lambda x: pos_check(x, 'adv'))
        df['pron_count'] = df['text'].apply(lambda x: pos_check(x, 'pron'))


        return df

    # Psychological features
    def psychological_features(self):
        df = self.data

        """
        Sentiment related features from the text

        
        Feature 8: Article Polarity - polarity score of essay
        Feature 9: Article Subjectivity - subjectivity score of essay

        """

        # Helper Functions
        def get_polarity(text):
            try:
                textblob = TextBlob(unicode(text, 'utf-8'))
                pol = textblob.sentiment.polarity
            except:
                pol = 0.0
            return pol

        def get_subjectivity(text):
            try:
                textblob = TextBlob(unicode(text, 'utf-8'))
                subj = textblob.sentiment.subjectivity
            except:
                subj = 0.0
            return subj

        # Extract grammatic featues
        df['polarity'] = df['text'].apply(get_polarity)
        df['subjectivity'] = df['text'].apply(get_subjectivity)

        return df
    


