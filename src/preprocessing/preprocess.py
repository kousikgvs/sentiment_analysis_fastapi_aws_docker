import re
def remove_html_tags(text):
    pattern = re.compile('<.*?>')
    return pattern.sub(r'', text)

def remove_url(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'', text)

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    words = text.split()
    filtered = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered)

chat_words  = {
    "A3": "Anytime, Anywhere, Anyplace",
    "ADIH": "Another Day In Hell",
    "AFK": "Away From Keyboard",
    "AFAIK": "As Far As I Know",
    "ASAP": "As Soon As Possible",
    "ASL": "Age, Sex, Location",
    "ATK": "At The Keyboard",
    "ATM": "At The Moment",
    "BAE": "Before Anyone Else",
    "BAK": "Back At Keyboard",
    "BBL": "Be Back Later",
    "BBS": "Be Back Soon",
    "BFN": "Bye For Now",
    "B4N": "Bye For Now",
    "BRB": "Be Right Back",
    "BRUH": "Bro",
    "BRT": "Be Right There",
    "BSAAW": "Big Smile And A Wink",
    "BTW": "By The Way",
    "BWL": "Bursting With Laughter",
    "CSL": "Can’t Stop Laughing",
    "CU": "See You",
    "CUL8R": "See You Later",
    "CYA": "See You",
    "DM": "Direct Message",
    "FAQ": "Frequently Asked Questions",
    "FC": "Fingers Crossed",
    "FIMH": "Forever In My Heart",
    "FOMO": "Fear Of Missing Out",
    "FR": "For Real",
    "FWIW": "For What It's Worth",
    "FYP": "For You Page",
    "FYI": "For Your Information",
    "G9": "Genius",
    "GAL": "Get A Life",
    "GG": "Good Game",
    "GMTA": "Great Minds Think Alike",
    "GN": "Good Night",
    "GOAT": "Greatest Of All Time",
    "GR8": "Great!",
    "HBD": "Happy Birthday",
    "IC": "I See",
    "ICQ": "I Seek You",
    "IDC": "I Don’t Care",
    "IDK": "I Don't Know",
    "IFYP": "I Feel Your Pain",
    "ILU": "I Love You",
    "ILY": "I Love You",
    "IMHO": "In My Honest/Humble Opinion",
    "IMU": "I Miss You",
    "IMO": "In My Opinion",
    "IOW": "In Other Words",
    "IRL": "In Real Life",
    "IYKYK": "If You Know, You Know",
    "JK": "Just Kidding",
    "KISS": "Keep It Simple, Stupid",
    "L": "Loss",
    "L8R": "Later",
    "LDR": "Long Distance Relationship",
    "LMK": "Let Me Know",
    "LMAO": "Laughing My A** Off",
    "LOL": "Laughing Out Loud",
    "LTNS": "Long Time No See",
    "M8": "Mate",
    "MFW": "My Face When",
    "MID": "Mediocre",
    "MRW": "My Reaction When",
    "MTE": "My Thoughts Exactly",
    "NVM": "Never Mind",
    "NRN": "No Reply Necessary",
    "NPC": "Non-Player Character",
    "OIC": "Oh I See",
    "OP": "Overpowered",
    "PITA": "Pain In The A**",
    "POV": "Point Of View",
    "PRT": "Party",
    "PRW": "Parents Are Watching",
    "ROFL": "Rolling On The Floor Laughing",
    "ROFLOL": "Rolling On The Floor Laughing Out Loud",
    "ROTFLMAO": "Rolling On The Floor Laughing My A** Off",
    "RN": "Right Now",
    "SK8": "Skate",
    "STATS": "Your Sex And Age",
    "SUS": "Suspicious",
    "TBH": "To Be Honest",
    "TFW": "That Feeling When",
    "THX": "Thank You",
    "TIME": "Tears In My Eyes",
    "TLDR": "Too Long, Didn’t Read",
    "TNTL": "Trying Not To Laugh",
    "TTFN": "Ta-Ta For Now!",
    "TTYL": "Talk To You Later",
    "U": "You",
    "U2": "You Too",
    "U4E": "Yours For Ever",
    "W": "Win",
    "W8": "Wait...",
    "WB": "Welcome Back",
    "WTF": "What The F**k",
    "WTG": "Way To Go!",
    "WUF": "Where Are You From?",
    "WYD": "What You Doing?",
    "WYWH": "Wish You Were Here",
    "ZZZ": "Sleeping, Bored, Tired"
}

def chat_conversion(text):
    new_text = []
    for w in text.split():
        if w.upper() in chat_words:
            new_text.append(chat_words[w.upper()])
        else:
            new_text.append(w)
    return " ".join(new_text)

from autocorrect import Speller

spell = Speller(lang='en')

def correct_spelling_autocorrect(text):
    return spell(text)

import emoji

def replace_emojis(text):
    return emoji.demojize(text)

import re

def clean_text(text):
    """
    Cleans text for sentiment analysis:
    - Removes decorative symbols: ----, (), [], {}, /
    - Keeps sentiment-carrying punctuation: !, ?, ...
    - Replaces multiple spaces with single space
    """
    # Remove brackets but keep content inside
    text = re.sub(r'\((.*?)\)', r'\1', text)
    text = re.sub(r'\[(.*?)\]', r'\1', text)
    text = re.sub(r'\{(.*?)\}', r'\1', text)

    # Remove slashes, dashes (sequences of - or /)
    text = re.sub(r'[-/]+', ' ', text)

    # Remove other non-alphanumeric characters except ! ? . , '
    # You can adjust to keep ',' if needed
    text = re.sub(r'[^a-zA-Z0-9\s!?.]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess(df):
    print("Started with Removing HTML Tags....")
    df["review"] = df["review"].apply(remove_html_tags)
    print("completed with Removing HTML Tags....")
    print("Started with Remove URL ....")
    df["review"] = df["review"].apply(remove_url)
    print("Completed with Remove URL ....")
    print("Started with Chat Conversion ....")
    df["review"] = df["review"].apply(chat_conversion)
    print("Completed with Chat Conversion ....")    
    print("Started with Emoji conversion ....")
    df["review"] = df["review"].apply(replace_emojis)
    print("Completed with Emoji conversion ....")
    print("Started with Cleaning Text ....")
    df["review"] = df["review"].apply(clean_text)
    print("Completed with Cleaning Text ....")    
    return df