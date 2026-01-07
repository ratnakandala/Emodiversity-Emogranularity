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


#Load the file with emotion words
emotion_words_df = pd.read_excel(r"PATH TO THE FILE")
#Specify the set of positive emotion words
positive_emotion_words_df = set(emotion_words_df['positive_emotion_word'].dropna().str.lower().str.strip())
#Specify the set of negative emotion words
negative_emotion_words_df = set(emotion_words_df['negative_emotion_word'].dropna().str.lower().str.strip())


#Load the data
dataset_path = #Specify the path to your dataset
#Load the dataset into a pandas DataFrame
dataset = pd.read_csv(dataset_path, encoding='utf-8-sig')
#Name of the column containing text data to process
text_column_name = 'replaced_lemmas'

#Load Dutch embedding model RobBERTa
#the 'base' version provides 768-dimensional vectors for each token/word
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
    

def compute_granularity(word_dict):
    #This function calculates the granularity using Cronbach's alpha
    #It's input argument can be a dict of embeddings (unique emotion words) or a list of embeddings (each item = an occurrence of an emotion word)
    
    #If the input is a dictionary (for unique words)
    if isinstance(word_dict, dict):
        #Convert the dictionary to a DataFrame
        #Each column represents a word, each row represents an embedding dimension
        df = pd.DataFrame(word_dict)

    #If the input is a list (multiple occurrences of words)
    elif isinstance(word_dict, list) and len(word_dict) > 0:
        #Convert the list of embeddings to a DataFrame
        #When the list of embeddings is called, you get each embedding dimension as a column and each word occurrence as a row
        #But Cronbach's alpha function in pingouin expects each column to represent a variable (word occurrence) 
                    #and each row to represent an observation (embedding dimension)
        #.T means "transpose", so that each column represents a word occurrence, each row represents an embedding dimension
        # Without this transpose, the function will be computing alpha across dimensions instead of words - which would be meaningless for granularity
        df = pd.DataFrame(word_dict).T
    else: #If it's empty or not the right type, return NaN and zero word count
        return np.nan, 0

    #Only compute the granularity if there is more than one word (column) in the DataFrame (atleast 2 words)
    if df.shape[1] > 1:
        alpha = pg.cronbach_alpha(data = df)
        granularity = 1 - alpha[0]
    else:
        granularity = np.nan
    return granularity, df.shape[1]


#Function to process a (single) text for the calculation of emotional granularity
def process_single_text(text):
    # initialize dictionaries to store aggregated (mean) embeddings for unique emotion words
    pos_word_avg, neg_word_avg, all_word_avg = {}, {}, {}

    #initialize lists to track the raw chronological sequence of emotion words found
    all_occurrences_pos_words, all_occurrences_neg_words, all_occurrences_emotion_words = [], [], []

    #intitialize lits to store every occurrence of emotion word embedding/vector (before averaging)
    all_pos_word_embeddings, all_neg_word_embeddings, all_emotion_word_embeddings = [], [], []

    #flair library is generally used for NLP tasks like generating word embeddings, NER, text classification, etc.
    #The line below converts the input text into a flair Sentence object, which is a data structure used by the flair library to represent and process text. (in a format that the library understands the input text string)
    text = flair.data.Sentence(text) #wrapping the raw string into a "Sentence" object that Flair understands.

    #Applying the Transformer (embedding) model to that "Sentence" object to generate tensors for every token in the sentence
    embedder.embed(text)

    #Iterate through each tokenized word in the text being processed
    for word in text:
        word_text = word.text.lower().strip() #Get the word text in lowercase and stripped of whitespace
        word_type = categorize(word_text) #Categorize the word as positive, negative, or neutral as per the emotion word lists defined previously
        #print(word_type)

        if word_type == 'positive':
            #Record the raw occurrence and the specific vector for this instance
            all_occurrences_pos_words.append(word_text) #Add every occurence of a positive emotion word to the overall positive words list
            all_occurrences_emotion_words.append(word_text) #Add the word to the overall emotion words list

            #Convert the PyTorch tensor to a standard Python list for processing
            emb = word.embedding.tolist() 
            all_pos_word_embeddings.append(emb) # Append the vector to the list of all positive word embeddings
            all_emotion_word_embeddings.append(emb) # Append the vector to the list of all word embeddings

            #Averaging embeddings for repeated positive emotion words
            if word_text in pos_word_avg: #if the word exists in the dict of the average embeddings for positive emotion words
                pos_word_avg[word_text] = np.mean( #update the mean using the new instance vector
                    [pos_word_avg[word_text], emb], axis = 0
                    )
            else:
                #If it's a new word, initialize its entry in the dict with its current embedding
                pos_word_avg[word_text] = emb
            
            
            #Update the 'Overall' dictionary using the same logic
            if word_text in all_word_avg:
                all_word_avg[word_text] = np.mean(
                    [all_word_avg[word_text], emb], axis = 0
                    )
            else:
                all_word_avg[word_text] = emb

        #Logic for negative emotion words (Parallel to positive words)
        elif word_type == 'negative':
            all_occurrences_neg_words.append(word_text)
            all_occurrences_emotion_words.append(word_text)

            emb = word.embedding.tolist() # Convert tensor to list
            all_neg_word_embeddings.append(emb) # Append to the list of all negative word embeddings
            all_emotion_word_embeddings.append(emb) # Append to the list of all word embeddings

            if word_text in neg_word_avg: #Store unique word averages for reference
                neg_word_avg[word_text] = np.mean([neg_word_avg[word_text], emb], axis = 0)
            else:
                neg_word_avg[word_text] = emb


            if word_text in all_word_avg:
                all_word_avg[word_text] = np.mean([all_word_avg[word_text], emb], axis = 0)
            else:
                all_word_avg[word_text] = emb


    #Grantularity calculations
    #Pass the dictionaries of averaged vectors to the Cronbach's alpha function ("compute_granularity" defined previously) to compute granularity
    #This computes (1 - alpha) value for each text
    pos_granularity, all_occurrences_pos_word_count = compute_granularity(pos_word_avg)
    neg_granularity, all_occurrences_neg_word_count = compute_granularity(neg_word_avg)    
    overall_granularity, overall_word_count = compute_granularity(all_word_avg)

    #Metadata Collection
    #Count how many unique emotion words were used in the text 
    #list already returns the unique emotion words
    positive_unique_emotion_words = list(pos_word_avg.keys())
    negative_unique_emotion_words = list(neg_word_avg.keys())
    all_unique_emotion_words = list(all_word_avg.keys())

    pos_unique_count = len(positive_unique_emotion_words)
    neg_unique_count = len(negative_unique_emotion_words)
    overall_unique_count = len(all_unique_emotion_words)
    
    # Output Generation
    # Return all the collected data as a pandas Series
    return pd.Series({
            'positive_all_words': all_occurrences_pos_words, #List of all positive emotion words found in the text
            'positive_all_words_count' : len(all_occurrences_pos_words), #Count of all the occurrences of positive emotion words
            'positive_unique_words': positive_unique_emotion_words, #Unique positive emotion words found in the text
            'positive_unique_wordcount': pos_unique_count, #Unique positive emotion word count
            'positive_granularity': pos_granularity, #Positive granularity score for the text

            'negative_all_words': all_occurrences_neg_words, #List of all negative emotion words found in the text
            'negative_all_words_count' : len(all_occurrences_neg_words), #Count of all the occurrences of negative emotion words
            'negative_unique_words': negative_unique_emotion_words, #Unique negative emotion words found in the text
            'negative_unique_wordcount': neg_unique_count, #Unique negative emotion word count
            'negative_granularity': neg_granularity, #Negative granularity score for the text

            'overall_words': all_occurrences_emotion_words, #List of all negative emotion words found in the text
            'overall_words_count' : len(all_occurrences_emotion_words), #Count of all the occurrences of negative emotion words
            'overall_unique_words': all_unique_emotion_words, #Unique negative emotion words found in the text
            'overall_unique_wordcount': overall_unique_count, #Unique negative emotion word count
            'overall_granularity': overall_granularity, #Negative granularity score for the text
        })

#Apply the processing function to each text in the dataset to compute text-level granularity scores
granularity_results = dataset[text_column_name].apply(process_single_text)

#Concatenate the granularity results with the original dataset
text_level_dataset = pd.concat([dataset, granularity_results], axis = 1)


# Person-level aggregation of text-level granularity scores
text_level_dataset['alias'] = text_level_dataset['alias'].astype(str).str.strip()

person_level_granularity = (
    text_level_dataset.groupby('alias', dropna=False)
    .agg(
        # mean granularity + affect
        positive_granularity_person_avg=('positive_granularity', 'mean'),
        negative_granularity_person_avg=('negative_granularity', 'mean'),
        overall_granularity_person_avg=('overall_granularity', 'mean'),
        valence_person_avg=('valence', 'mean'),
        arousal_person_avg=('arousal', 'mean'),

        # emotion word counts
        # all words: sum and avg (per prompt)
        positive_all_words_count_person_sum=('positive_all_words_count  ', 'sum'),
        positive_all_words_count_person_avg=('positive_all_words_count', 'mean'),

        negative_all_words_count_person_sum=('negative_all_words_count', 'sum'),
        negative_all_words_count_person_avg=('negative_all_words_count', 'mean'),
        overall_words_count_person_sum=('overall_words_count', 'sum'),
        overall_words_count_person_avg=('overall_words_count', 'mean'),

        # unique words: only sum
        positive_unique_wordcount_person_sum=('positive_unique_wordcount', 'sum'),
        negative_unique_wordcount_person_sum=('negative_unique_wordcount', 'sum'),
        overall_unique_wordcount_person_sum=('overall_unique_wordcount', 'sum'),

        # combine word lists
        positive_all_words=('positive_all_words', lambda x: [w for sublist in x for w in sublist]),
        positive_unique_words=('positive_unique_words', lambda x: list(set([w for sublist in x for w in sublist]))),

        negative_all_words=('negative_all_words', lambda x: [w for sublist in x for w in sublist]),
        negative_unique_words=('negative_unique_words', lambda x: list(set([w for sublist in x for w in sublist]))),

        overall_words=('overall_words', lambda x: [w for sublist in x for w in sublist]),
        overall_unique_words=('overall_unique_words', lambda x: list(set([w for sublist in x for w in sublist])))
    )
    .reset_index()
)

# Merge back to text-level dataset
cols_to_add = [c for c in person_level_granularity.columns if c not in text_level_dataset.columns or c == 'alias']
text_level_dataset = text_level_dataset.merge(
    person_level_granularity[cols_to_add],
    on='alias',
    how='left'
)


#Save person-level granularity dataset
person_level_granularity.to_csv(
    r"Specify the path to save the person-level granularity dataset",
    index=False,
    encoding='utf-8-sig'
)

print("✅ Saved person-level granularity output successfully.")

#Impute one-word texts with person-level averages
#Identify rows where exactly one positive emotion word was detected
one_positive_emotion_word = text_level_dataset['positive_all_words_count'] == 1

#Create a boolean mask to target the specific rows for imputation
m_pos = (
    one_positive_emotion_word &  # only exactly one positive emotion word
    (text_level_dataset['overall_words_count'] > 0) &  # exclude 0-word texts
    text_level_dataset['positive_granularity'].isna()  # only if NaN
)


# Imputation Logic: For single-word occurrences, use the participant's average
# This replaces the NaN value for those texts with the 'person-level' average positive granularity value
text_level_dataset.loc[m_pos, 'positive_granularity'] = (
    text_level_dataset.loc[m_pos, 'positive_granularity_person_avg']
)

#Repeat the same process for negative one-word texts
one_negative_emotion_word = text_level_dataset['negative_all_words_count'] == 1

m_neg = (
    one_negative_emotion_word &  # only exactly one negative emotion word
    (text_level_dataset['overall_words_count'] > 0) &  # exclude 0-word texts
    text_level_dataset['negative_granularity'].isna()  # only if NaN
)

# Fill negative granularity for these rows
text_level_dataset.loc[m_neg, 'negative_granularity'] = (
    text_level_dataset.loc[m_neg, 'negative_granularity_person_avg']
)

print(
    f"Filled one-word rows — positive: {m_pos.sum()}, negative: {m_neg.sum()}"
)

#Save text-level granularity dataset
text_level_dataset.to_csv(
    r"#Specify the path to save the text-level granularity dataset",
    index=False,
    encoding='utf-8-sig'
)

print("✅ Saved text-level granularity output successfully.")
