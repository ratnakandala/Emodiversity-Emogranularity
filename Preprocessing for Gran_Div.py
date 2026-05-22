# import modules
import csv
import re
import pandas as pd
import math
import string
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from wordcloud import WordCloud
from matplotlib.backends.backend_pdf import PdfPages

import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
nltk.download('punkt_tab')
import torch
#Checking if cuda is available
is_cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {is_cuda_available}")



#STEP 1: DATA CLEANING
with open(r'input_data.csv', 'r', encoding='utf-8-sig') as fr,\
        open('Data_Cleaning_output.csv', 'w', encoding='utf-8-sig', newline='') as fw:
    reader = csv.reader(fr)
    writer = csv.writer(fw)

    next(fr)
    header = ['connectionId', 'alias', 'sentBeepId', 'timeStampStart', 'typed', 'valence', 'arousal', 'location',
              'steps', 'noise', 'text','regex_output']
    writer.writerow(header)

    for row in reader:
        # specify columns to use
        connectionID = row[0]
        alias = row[3]
        beepID = row[6]
        timeStamp = row[11]
        typed = row[18]
        valence = row[19]
        arousal = row[20]
        location = row[21]
        steps = row[24]
        noise = row[25]
        text = row[26]

        if text.strip() == '' or text == "NaN" or text.strip() == "<sos/eos>": #Remove NaN rows
            continue
        
        text = ''.join(text.splitlines())

        #Remove the annotation tags and run the cleaned_text through the nlp pipeline
        patterns = r'\*[v,d,z,u,p]|\<spk\>|ggg|xxx|\w*\*[a,x]'
        cleaned_text = re.sub(patterns, '', text).strip()
               
        to_write = [connectionID, alias, beepID, timeStamp, typed, valence, arousal, location, steps, noise, text, cleaned_text]
        writer.writerow(to_write)


#STEP 2: NAMES REMOVAL

names_to_remove = [
    "Katie", "katie", "Cathy", "Kathy", "Kat", "Kaity", "Kadie", "Karie", "Carrie", "Heidi", "Heiti", "Cazy",
    "Kathie", "Kenny", "Kitty", "kety", "kickie", "Katrien", "Jetty", "Kelly", "Terry",
    "Carine", "kettie", "ketie", "Kedi", "Ketie", "Kevin", "Kris", "Katia", "Ketje",
    "Kesi", "feti", "Kristel", "Kimke", "Chetie", "Ketia", "Kate", "Kizi", "kijk", "Kim",
    "kijkt", "kick", "Kindy", "kethy", "ke die", "kity", "Kiki", "Kady", "kadi", "kidi",
    "cany", "kike", "Keti", "Ketty",  "ketty", "Cato", "Kaa", "Kaat", "Kaats", "Greet", "Kaatje", "yoke diep", "kiki"
]

joined_names = '|'.join(sorted(names_to_remove, key=len, reverse=True))


# Create the regex pattern
patterns = (
    r'(?:^|(?<![a-zA-Z]))(' + joined_names + r')(?!=[a-zA-Z])'   # prefix match: Katie in Katieik
    r'|'                                                      
    r'(?<=[a-zA-Z])(' + joined_names + r')(?![a-zA-Z])'        # suffix match: kickie in hekickie
    r'|'                                                      
    r'(?<![a-zA-Z])(' + joined_names + r')(?![a-zA-Z])'        # exact standalone name
)

data = #specify path to Data_Cleaning_output.csv; column name: 'regex_output'

# Apply regex to remove names
data['name_reference_removal_output'] = data['regex_output'].apply(
    lambda text: re.sub(patterns, '', text).strip()
)

#Removal of greetings in the first ten words
with open(r"greetings.txt", "r", encoding="utf-8") as file:
    greetings = [line.strip() for line in file if line.strip()]  # Remove empty lines and strip whitespace

# Compile the greetings into a regex pattern
pattern = r"(?<!\w)(" + '|'.join(map(re.escape, greetings)) + r")\b[\.,!?]*"
patterns = re.compile(pattern, re.IGNORECASE)

n = 10  # word_limit

def clean_ini_words(text, word_limit=n):
    words = text.split()  # Split the text into words based on whitespace
    if not words:
        return text
    split_point = min(word_limit, len(words))
    # Process the first 'word_limit' words
    first_part = ' '.join(words[:split_point])
    # Remove greetings from the first part
    first_part_cleaned = patterns.sub("", first_part).strip()
    # Split the cleaned part into words
    cleaned_words = first_part_cleaned.split()
    # Combine cleaned words with the remaining words
    combined_words = cleaned_words + words[split_point:]
    cleaned_text = ' '.join(combined_words)
    # Fix punctuation spacing issues
    cleaned_text = re.sub(r'(\s)([.!?])', r'\2', cleaned_text)  # Remove space before punctuation
    cleaned_text = re.sub(r'([.!?])(\S)', r'\1 \2', cleaned_text)  # Add space after punctuation if needed
    return cleaned_text

# Clean each text in the list
#Handle None values too
data['spellcheck_output'] = data['name_reference_removal_output'].apply(lambda text: clean_ini_words(text, word_limit = n) if isinstance(text, str) else text)


# Save the DataFrame with the new 'cleaned_text' column to a new CSV file
output_file_path = "Spellcheck_output.csv"
data.to_csv(output_file_path, index=False, encoding="utf-8-sig") 


#STEP3: LEMMATIZATION
nlp = stanza.Pipeline(lang = 'nl', processors = 'tokenize,pos,lemma', download_method=None, use_gpu = is_cuda_available)

with open(r'Spellcheck_output.csv', 'r', encoding='utf-8-sig') as fr,\
        open('Lemmatization_output_before_Conversion_List.csv', 'w', encoding='utf-8-sig', newline='') as fw:
    reader = csv.reader(fr)
    writer = csv.writer(fw)

    next(fr)
    header = ['connectionId', 'alias', 'sentBeepId', 'timeStampStart', 'typed', 'valence', 'arousal', 'location',
              'steps', 'noise', 'text', 'Spellcheck_Output', 'length', 'sentences', 'lemmas']
    writer.writerow(header)

    for row in reader:
        # specify columns to use
        connectionID = row[0]
        alias = row[1]
        beepID = row[2]
        timeStamp = row[3]
        typed = row[4]
        valence = row[5]
        arousal = row[6]
        location = row[7]
        steps = row[8]
        noise = row[9]
        text = row[10]
        spellcheck_output = row[13]
        
        #run the spellcheck_ouput through the nlp pipeline
        doc = nlp(spellcheck_output) #'doc' object contains all the info

        # get length descriptives
        sentences = sent_tokenize(spellcheck_output)
        num_sents = len(sentences)
        text_no_punct = spellcheck_output.translate(str.maketrans('', '', string.punctuation))
        words = word_tokenize(text_no_punct)
        num_words = len(words)

        # lemmatize text
        lemmas = [word.lemma for sentence in doc.sentences for word in sentence.words]
        cleaned_lemmas = ' '.join(str(lemma) for lemma in lemmas)

        to_write = [connectionID, alias, beepID, timeStamp, typed, valence, arousal, location, steps, noise, text, spellcheck_output,
                    num_words, num_sents, cleaned_lemmas]
        writer.writerow(to_write)
    
data = #path to the file "Lemmatization_output_before_Conversion_List.csv"

#Run the conversion list again on the lemmatization output
#Replace the words from the Conversion List
Conversions = pd.read_csv(r"conversion list.csv", header = None) #Read the csv file with the conversion list
original_word = Conversions.iloc[:,0].astype(str).str.strip() #Extract the first column as Series
replaced_word = Conversions.iloc[:,1].astype(str).str.strip() #Extract the second column as Series
#Create a dictionary (key-value pairs) to words the lemmas
replace_dict = pd.Series(replaced_word.values, index = original_word.values).to_dict()

# Create a pattern with word boundaries to ensure exact word matches
#k,v : key, value
replace_dict_with_boundaries= {
    r'(?i)\b' + re.escape(k) + r'\b': v for k, v in replace_dict.items()
}

# Replace the words in the original data file using regex for exact word matching
data['replaced_lemmas'] = data['lemmas'].replace(replace_dict_with_boundaries, regex=True) #lemmas after replacements #Keys in the dict to be interpreted as re

#Save the lemmatization output after running the conversion list
output = 'Lemmatization_output.csv'
data.to_csv(output, index=False, encoding="utf-8-sig")