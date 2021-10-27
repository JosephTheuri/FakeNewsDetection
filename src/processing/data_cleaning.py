import re


class DataCleaning():
    def __init__(self, data, language):
        self.data = data
        self.language = language

    # Text to lowercase


    # Remove any url
    def remove_URL(text):
        return re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove any HTML tags
    def remove_html(text):
        html = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
        return re.sub(html, "", text)

    # Remove Non-ASCI
    def remove_non_ascii(text):
        return re.sub(r'[^\x00-\x7f]',r'', text)

    # Remove Special Characters
    def remove_special_characters(text):
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




