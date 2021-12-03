import string
import pandas as pd
from sklearn.utils.extmath import density
from textblob import TextBlob
from nltk.corpus import stopwords
# from data_cleaning import DataCleaning

class FeatureExtraction():
    def __init__(self, data, language):
        self.data = data
        # self.text = DataCleaning(data = data['text']).clean_data()
        self.text =  data['text']
        self.language = language

    # Text based features
    def text_features(self):
        text = self.text
        df = pd.DataFrame()

        """
       
        Feature 1: Word density - average word count per sentence
        Feature 2: Number_density - propotion of numbers to total words in the essay
        Feature 3: Type-Token-Ratio - proportion of unique words
        Feature 4: Puncutation density in the Complete Essay - propotion of punctuation marks to total characters in the essay
        Feature 5: Upper Case density in the Complete Essay - proportion of upper count words to total words in the essay
        Feature 6: Title Word density in the Complete Essay - proportion of of proper case (title) words to total words in the essay
        Feature 7: Stopword density in the Complete Essay - proportion of of stopwords to total words in the essay
       
        """

        # Helper functions
        punctuation = string.punctuation
        stop_words = list(set(stopwords.words(self.language)))

        # Extract features from text
        char_count = text.apply(len)    #  Count of characters
        word_count = text.apply(lambda x: len(x.split()))   # Count of words
        sentence_count = text.apply(lambda x: len(x.split('.')))   # Count of sentences

        df['word_density'] = word_count / sentence_count    # Calculate word density
        df['number_density'] = text.apply(lambda x: len([nb for nb in x.split() if any(x.isdigit() for x in nb)]))/word_count    # Calculate number density
        df['ttr'] = text.apply(lambda x: len(set(x.split())))/word_count    # Calculate Type token ratio
        df['punctuation_density'] = text.apply(lambda x: len("".join(_ for _ in x if _ in punctuation)))/char_count    # Calculate punctuation count
        df['title_word_density'] = text.apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))/word_count   # Calculate title case count
        df['upper_case_word_density'] = text.apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))/word_count  # Calculate upper case count
        df['stopword_density'] = text.apply(lambda x: len([wrd for wrd in x.split() if wrd.lower() in stop_words]))/word_count # Calculate stop word case count

        return df

    # Psychological features
    def psychological_features(self):
        text = self.text
        df = pd.DataFrame()

        """
        Sentiment related features from the text

        Feature 8: Article Polarity - polarity score of essay
        Feature 9: Article Subjectivity - subjectivity score of essay

        """

        # Helper Functions
        def get_polarity(x):
            try:
                textblob = TextBlob(unicode(x, 'utf-8'))
                pol = textblob.sentiment.polarity
            except:
                pol = 0.0
            return pol

        def get_subjectivity(x):
            try:
                textblob = TextBlob(unicode(x, 'utf-8'))
                subj = textblob.sentiment.subjectivity
            except:
                subj = 0.0
            return subj

        # Extract grammatic featues
        df['polarity_'] = text.apply(get_polarity)
        df['subjectivity_'] = text.apply(get_subjectivity)

        return df
    
    # Grammatic features
    def grammatic_features(self):
        text = self.text
        df = pd.DataFrame()

        """
        Part of Speech related features from the text

        Feature 10: Noun Count - proportion of nouns in essay
        Feature 11: Verb Count - proportion of verbs in essay
        Feature 12: Adjective Count - proportion of adjectives in essay
        Feature 13: Adverb Count - proportion of adverbs in essay
        Feature 14: Pronoun Count - proportion of pronouns in essay
        Feature 15: Preposition Count - proportion of prepositions in essay
        Feature 16: Interjections Count - proportion of interjections in essay
        Feature 17: Determiners Count - proportion of determiners in essay
        Feature 18: Conjunction Count - proportion of conjunctions in essay
        Feature 19: Modal Count - proportion of modals in essay
        Feature 20: Index of Formality - F = (noun + adjective + preposition – pronoun – verbs – participles – adverbs – interjections + 100)/2
        Feature 21: Index of the lexical density - ratio  of  function  words  to  content words
        Feature 22: POS Diversity - proportion on unique POS tags to total tags

        """

        # Helper Functions
        ## Specify POS dict
        pos_dic = {
            'noun' : ['NN','NNS','NNP','NNPS'],
            'pron' : ['PRP','PRP$','WP','WP$'],
            'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
            'adj' :  ['JJ','JJR','JJS'],
            'adv' : ['RB','RBR','RBS','WRB'],
            'prep' : ['IN'],    # Preposition
            'int': ['UH'],    # Interjections
            'dt': ['DT'],    # Determiners
            'cc': ['CC', 'CD'],    # Conjunction
            'md': ['MD']    # Modal
        }

        ## Check and get the part of speech tag count of a words in a given sentence
        def pos_check(x):
            try:
                wiki = TextBlob(x)
                tags = [x[1] for x in wiki.tags]
                tag_count = pd.Series(tags).value_counts().to_dict()

                cnt = {
                'noun': sum([tag_count[k] for k in tag_count.keys() if k in pos_dic['noun']]), 
                'pron': sum([tag_count[k] for k in tag_count.keys() if k in pos_dic['pron']]), 
                'verb': sum([tag_count[k] for k in tag_count.keys() if k in pos_dic['verb']]), 
                'adj': sum([tag_count[k] for k in tag_count.keys() if k in pos_dic['adj']]), 
                'adv': sum([tag_count[k] for k in tag_count.keys() if k in pos_dic['adv']]),
                'prep': sum([tag_count[k] for k in tag_count.keys() if k in pos_dic['prep']]),
                'int': sum([tag_count[k] for k in tag_count.keys() if k in pos_dic['int']]),
                'dt': sum([tag_count[k] for k in tag_count.keys() if k in pos_dic['dt']]),
                'cc': sum([tag_count[k] for k in tag_count.keys() if k in pos_dic['cc']]),
                'md': sum([tag_count[k] for k in tag_count.keys() if k in pos_dic['md']]),
                }

                density = {k: v/sum(tag_count.values())*100 for k, v in cnt.items()}
                
                # index  of  formality: F = (noun + adjective + preposition – pronoun – verbs – participles – adverbs – interjections + 100)/2
                iof = (cnt['noun'] + cnt['adj'] + cnt['prep'] - cnt['pron'] - cnt['verb'] - cnt['adv'] - cnt['int'] + 100)/2

                # index of the lexical density: ratio  of  function  words  to  content words
                iol = (cnt['pron']+cnt['prep']+cnt['int']+cnt['dt']+cnt['cc']+cnt['md'])/(cnt['noun']+cnt['verb']+cnt['adj']+cnt['adv'])

                # POS diversity
                pos_diversity = len(set(tags))/len(tags)

            except:
                density = {'noun':0, 'pron':0, 'verb':0, 'adj':0, 'adv':0, 'prep':0, 'int':0, 'dt':0, 'cc':0, 'md':0}
                iof = 0
                iol = 0
                pos_diversity = 1
                pass
            return (density, iof, iol, pos_diversity)
            # return density

        # Extract grammatic featues
        
        output = text.apply(lambda x: pos_check(x))
        density = [x[0] for x in output]
        iof = [x[1] for x in output]
        iol = [x[2] for x in output]
        pos_diversity = [x[3] for x in output]
        
        df['noun_density'] = [x['noun'] for x in density]
        df['verb_density'] = [x['verb'] for x in density]
        df['adj_density'] = [x['adj'] for x in density]
        df['adv_density'] = [x['adv'] for x in density]
        df['pron_density'] = [x['pron'] for x in density]
        df['prep_density'] = [x['prep'] for x in density]
        df['int_density'] = [x['int'] for x in density]
        df['dt_density'] = [x['dt'] for x in density]
        df['cc_density'] = [x['cc'] for x in density]
        df['md_density'] = [x['md'] for x in density]
        df['iof_'] = iof
        df['iol_'] = iol
        df['pos_diversity'] = pos_diversity


        return df

    def extract_features(self):
        df = self.data
        print('\nFeature Extraction\n')

        print("Extracting Text features ...")
        df1 = self.text_features()
        print(" Done")

        print("\nExtracting Psychological features ...")
        df2 = self.psychological_features()
        print(" Done")

        print("\nExtracting Grammatic features ...")
        df3 = self.grammatic_features()
        print(" Done")

        output = pd.concat([df, 
        df1, 
        df2, 
        df3
        ], axis=1).fillna(0)
        print('\n', output.describe(include='all'), '\n')

        print('\nFeature Extraction Complete\n\n', '-----'*10)

        return output

