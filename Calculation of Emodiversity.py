import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import transformers
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import math
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


#emodiversity at the moment level #text level (a tree)
def shannon_index(text: str, emotion_list: list) -> float:
    """
        text: input string
        emotion_list: emotion_words
    """
    words = text.lower().split()
    word_counts = Counter(words)

    #Find which emotion words from the lists are in the texts and get their counts
    emotion_word_counts = [word_counts[emotion.lower()] for emotion in emotion_list if word_counts[emotion.lower()] > 0]

    #If no emotion words found, diversity will be 0
    if len(emotion_word_counts) <= 1:
        return 0.0

    total_emotions = sum(emotion_word_counts)
    proportions = [count / total_emotions for count in emotion_word_counts]
    shannon_index_value = -sum(p * math.log(p) for p in proportions)

    return shannon_index_value

#calculate emodiversity scores
def get_emodiversity_scores(text: str, positive_emotions: list, negative_emotions: list) -> dict:
    """
    Calculates positive and negative emodiversity for a single text.

    Args:
        text: The input string to analyze.
    
    Returns:
     A dictionary containing the positive and negative diversity scores
        e.g., {positive_diversity: 0.6, negative diversity: 0.4}
    """
    words = text.lower().split()
    word_counts = Counter(words)

    #Positive emotion counts
    positive_emotion_words_in_text = [word for word in positive_emotions if word in word_counts]
    positive_counts = sum(word_counts[word] for word in positive_emotion_words_in_text)
    positive_unique = len(positive_emotion_words_in_text)

    #Negative emotion counts
    negative_emotion_words_in_text = [word for word in negative_emotions if word in word_counts]
    negative_counts = sum(word_counts[word] for word in negative_emotion_words_in_text)
    negative_unique = len(negative_emotion_words_in_text)

    #Calculate diversity for positive emotions
    positive_diversity = shannon_index(text, positive_emotions)

    #Calculate diversity for negative emotions
    negative_diversity = shannon_index(text, negative_emotions)

    return{
        'positive_emotion_lemma_count': positive_counts,        #Total occurrences of positive emotion words including repetitions
        'positive_emotion_unique_count': positive_unique,
        'unique positive emotion lemmas': positive_emotion_words_in_text,
        'positive_diversity': positive_diversity,
        'negative_emotion_lemma_count': negative_counts,
        'negative_emotion_unique_count': negative_unique, 
        'unique negative emotion lemmas': negative_emotion_words_in_text,        
        'negative_diversity': negative_diversity
    }


input_texts = dataset['replaced_lemmas'].tolist()

#Call the combined function to calculate emodiversity scores
diversity_scores = [
    get_emodiversity_scores(text, positive_emotion_words_df, negative_emotion_words_df)
    for text in input_texts
    ]

diversity_scores_df = pd.DataFrame(diversity_scores)
final_dataset = dataset.merge(diversity_scores_df, left_index = True, right_index = True)
final_dataset.to_csv("Shannons_lemmatized_text_level.csv", index= False, encoding='utf-8-sig')


#Calculation of person level emodiversity
#Concatenate all texts per participant
participant_texts = final_dataset.groupby('alias')['text'].apply(lambda x: ' '.join(x)).reset_index()

#Apply the emodiversity function to the concatenated texts
participant_diversity = participant_texts['text'].apply(
    lambda x: get_emodiversity_scores(x, positive_emotion_words_df, negative_emotion_words_df)
).apply(pd.Series)

#Flatten the columns names
participant_diversity.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in participant_diversity.columns.values]

#Convert the 'alias' index back into a regular column
participant_diversity['alias'] = participant_texts['alias'].values
participant_diversity = participant_diversity.reset_index()

participant_diversity.to_csv("Shannons Scores_Person level.csv", index = False, encoding = 'utf-8-sig')

