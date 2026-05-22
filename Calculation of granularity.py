#Importing libraries for data handling and visualization
import pandas as pd #Import pandas for data manipulation and analysis
import matplotlib.pyplot as plt #Import matplotlib for plotting graphs
import numpy as np #Import numpy for numerical operations
import flair #Import flair for natural language processing tasks
from nltk.stem.snowball import SnowballStemmer #Import SnowballStemmer for stemming words
import pingouin as pg #Import pingouin for statistical analysis

import torch #Import torch for deep learning tasks
from flair.embeddings import TransformerWordEmbeddings #Import TransformerWordEmbeddings from flair for word embeddings
import transformers #Import transformers forfrom collections import Counter

import seaborn as sns #Import seaborn for advanced data visualization #heat maps, etc.

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #Set device to GPU if available, otherwise CPU
torch.cuda.is_available() #Check if CUDA is available for GPU computations

from itertools import chain
from collections import Counter
import string
import re

#Load the emotion words revised
emotion_words_df = pd.read_excel(r"Path to emotion words file")
positive_emotion_words_df = set(emotion_words_df['positive_word_lemma'].dropna().str.lower().str.strip())
negative_emotion_words_df = set(emotion_words_df['negative_word_lemma'].dropna().str.lower().str.strip())

#Load the lemmatized data
dataset_path = r"Path to the lemmatized data"
#Load the dataset into a pandas DataFrame
dataset = pd.read_csv(dataset_path, encoding='utf-8-sig') #Specify encoding to handle special characters
#Name of the column containing text data to process
text_column_name = 'replaced_lemmas'

#Total number of texts before filtering
total_texts = len(dataset)

#Filter out texts with <= 15 words (length column)
filtered_dataset = dataset[dataset['length'] >= 15]
filtered_texts = len(filtered_dataset)

#Load Dutch embedding model RobBERTa
embedder = flair.embeddings.TransformerWordEmbeddings(
    'pdelobelle/robbert-v2-dutch-base',
    trust_remote_code = True
)

#Categorize a given word as positive, negative, or neutral
def categorize(word):
    if word in positive_emotion_words_df:
        return 'positive'
    elif word in negative_emotion_words_df:
        return 'negative'
    else:
        return 'neutral'

def compute_granularity(word):
    #This function calculates the granularity using Cronbach's alpha
    #It's input argument can be a dict of embeddings (unique emotion words) or a list of embeddings (each item = an occurrence of an emotion word)
    
    #If the input is a dictionary (for unique words)
    if isinstance(word, dict):
        #Convert the dictionary to a DataFrame
        #Each column represents a word, each row represents an embedding dimension
        df = pd.DataFrame(word)

    #If the input is a list (multiple occurrences of words)
    elif isinstance(word, list) and len(word) > 0:
        #Convert the list of embeddings to a DataFrame
        #When the list of embeddings is called, you get each embedding dimension as a column and each word occurrence as a row
        #But Cronbach's alpha function in pingouin expects each column to represent a variable (word occurrence) 
                    #and each row to represent an observation (embedding dimension)
        #.T means "transpose", so that each column represents a word occurrence, each row represents an embedding dimension
        # Without this transpose, the function will be computing alpha across dimensions instead of words - which would be meaningless for granularity
        df = pd.DataFrame(word).T
    else: #If it's empty or not the right type, return NaN and zero word count
        return np.nan, 0

    #Only compute the granularity if there is more than one word (column) in the DataFrame (atleast 2 words)
    if df.shape[1] > 1:
        alpha = pg.cronbach_alpha(data = df)
        granularity = 1 - alpha[0]
    else:
        granularity = np.nan
    return granularity, df.shape[1]


#CODE BEING USED TO SAVE THE VARIABLE DETAILS
#POS: POSITIVE EMOTION WORDS
#NEG: NEGATIVE EMOTION WORDS
#UNIQUE: FOR UNIQUE EMOTION WORD COUNTING
#OCCURRENCES: INDICATES WE ARE CALCULATING EVERY OCCURRENCE OF AN EMOTION WORD

#Function to process single text for the calculation of emotional granularity
def process_single_text(text):

    # Pre-process: replace '/' with space so slash-joined words are split
    # e.g., "geïnteresseerd/nieuwsgierig" → "geïnteresseerd nieuwsgierig"
    text = text.replace('/', ' ')

    # dictionaries to hold average embeddings for each word
    pos_word_avg, neg_word_avg = {}, {}

    #lists to keep every occurrence of emotion words
    all_occurrences_pos_words, all_occurrences_neg_words= [], []

    #store every occurrence of emotion word embeddings
    all_pos_word_embeddings, all_neg_word_embeddings,  = [], []

    #flair library is generally used for NLP tasks like generating word embeddings, NER, text classification, etc.
    #The line below converts the input text into a flair Sentence object, which is a data structure used by the flair library to represent and process text. (in a format that the library understands the input text string)
    text = flair.data.Sentence(text, use_tokenizer=False) #wrapping the raw string into a "Sentence" object that Flair understands.

    #Applying the embedding model to that "Sentence" object
    embedder.embed(text)

    for word in text:
        word_text = re.sub(r'^[-\W]+|[-\W]+$', '', word.text.lower().strip()) #Get the word text in lowercase and stripped of whitespace
        word_type = categorize(word_text) #Categorize the word as positive, negative, or neutral as per the emotion word lists defined previously

        if word_type == 'positive':
            all_occurrences_pos_words.append(word_text) #Add every occurence of a positive emotion word to the overall positive words list

            #Saving the embeddings for the positive emotion words
            emb = word.embedding.tolist() # Convert tensor to list
            all_pos_word_embeddings.append(emb) # Append to the list of all positive word embeddings

            #Averaging embeddings for repeated positive emotion words
            if word_text in pos_word_avg:
                pos_word_avg[word_text] = np.mean(
                    [pos_word_avg[word_text], emb], axis = 0
                    )
            else:
                pos_word_avg[word_text] = emb

      
        elif word_type == 'negative':
            all_occurrences_neg_words.append(word_text)

            emb = word.embedding.tolist() # Convert tensor to list
            all_neg_word_embeddings.append(emb) # Append to the list of all negative word embeddings

            if word_text in neg_word_avg: #Store unique word averages for reference
                neg_word_avg[word_text] = np.mean([neg_word_avg[word_text], emb], axis = 0)
            else:
                neg_word_avg[word_text] = emb

    #Compute granularities using *all occurrences* of emotion words
    pos_granularity, _ = compute_granularity(all_pos_word_embeddings)
    neg_granularity, _ = compute_granularity(all_neg_word_embeddings)    

    #Extract unique emotion words found in the text #list already returns the unique emotion words [ONLY FOR SUMMARY NOT GRANULARITY CAL]
    positive_unique_emotion_words = list(pos_word_avg.keys())
    negative_unique_emotion_words = list(neg_word_avg.keys())

 
    pos_unique_count = len(positive_unique_emotion_words)
    neg_unique_count = len(negative_unique_emotion_words)

    
    return pd.Series({
            'positive_all_words_lemm': all_occurrences_pos_words, #List of all positive emotion words found in the text
            'positive_all_words_count_lemm' : len(all_occurrences_pos_words), #Count of all the occurrences of positive emotion words
            'positive_unique_words_lemm': positive_unique_emotion_words, #Unique positive emotion words found in the text
            'positive_unique_wordcount_lemm': pos_unique_count, #Unique positive emotion word count
            'positive_granularity_lemm': pos_granularity, #Positive granularity score for the text

            'negative_all_words_lemm': all_occurrences_neg_words, #List of all negative emotion words found in the text
            'negative_all_words_count_lemm' : len(all_occurrences_neg_words), #Count of all the occurrences of negative emotion words
            'negative_unique_words_lemm': negative_unique_emotion_words, #Unique negative emotion words found in the text
            'negative_unique_wordcount_lemm': neg_unique_count, #Unique negative emotion word count
            'negative_granularity_lemm': neg_granularity #Negative granularity score for the text
        })


granularity_results = filtered_dataset[text_column_name].apply(process_single_text)

text_level_initial_dataset = pd.concat([filtered_dataset, granularity_results], axis = 1)

text_level_initial_dataset.to_csv(
    r"Granularity Output Text Level.csv",
    index=False,
    encoding='utf-8-sig'
)





