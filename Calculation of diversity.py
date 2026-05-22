#IMPORT LIBRARIES
import pandas as pd #Import pandas library for data manipulation and visualization
import numpy as np #Import numpy library for numerical operations
import matplotlib.pyplot as plt #Import matplotlib library for data visualization

import re
import math
from collections import Counter
import string

#Load the lemmatized emotion words file
emotion_words_df = pd.read_excel(r"Path to emotion words file")

#Extract the unique positive emotion words (Because there are duplicates) from the dataframe, convert to lowercase and strip any leading/trailing whitespace
positive_emotion_lemmas = set(emotion_words_df['positive_word_lemma'].dropna().str.lower().str.strip())
negative_emotion_lemmas = set(emotion_words_df['negative_word_lemma'].dropna().str.lower().str.strip())

#Load the lemmatized data
dataset_path = r"Path to the lemmatized data"
#Load the dataset into a pandas DataFrame
dataset = pd.read_csv(dataset_path, encoding='utf-8-sig') #Specify encoding to handle special characters
#Name of the column containing text data to process
text_column_name = 'replaced_lemmas'

#Total number of texts - before filtering 
total_texts = len(dataset)

#Filter out texts with length <= 15 words
filtered_dataset = dataset[dataset['length'] >= 15]
filtered_texts = len(filtered_dataset)

input_texts = filtered_dataset['replaced_lemmas'].tolist()

#Calculate positive and negative emodiversity scores using the Shannon's index

#emodiversity at the moment level #text level (a tree)
def shannon_index(text: str, emotion_list: list) -> float:
    """
        text: input string
        emotion_list: emotion_words
    """
    text = text.replace('/', ' ').replace('\\', ' ') #for cases where emotion words have been written as a/b etc.

    words = text.lower().split()
    word_counts = Counter(
                    re.sub(r'^[-\W]+|[-\W]+$', '', w)  # strip leading/trailing non-word chars including -
                    for w in words)

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
    text = text.replace('/', ' ').replace('\\', ' ')
    
    words = text.lower().split()
    word_counts = Counter(
                    re.sub(r'^[-\W]+|[-\W]+$', '', w)  # strip leading/trailing non-word chars including -
                    for w in words)

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
        'positive_emotion_all_lemma_count_Shannons': positive_counts,        #Total occurrences of positive emotion words including repetitions
        'positive_emotion_unique_count_Shannons': positive_unique,
        'unique positive emotion lemmas_Shannons': positive_emotion_words_in_text,
        'positive_diversity_Shannons': positive_diversity,
        'negative_emotion_all_lemma_count_Shannons': negative_counts,
        'negative_emotion_unique_count_Shannons': negative_unique,
        'unique negative emotion lemmas_Shannons': negative_emotion_words_in_text,
        'negative_diversity_Shannons': negative_diversity
    }

#Call the function for each text (filtered) to calculate diversity scores
diversity_scores = [
            get_emodiversity_scores(text, positive_emotion_lemmas, negative_emotion_lemmas)
            for text in input_texts]


#Make a dataframe of the diversity scores
diversity_scores_df = pd.DataFrame(diversity_scores)

#Merge the diversity scores dataframe with the original dataset (FILTERED) based on the index
final_dataset = filtered_dataset.merge(diversity_scores_df, left_index = True, right_index = True)

final_dataset.to_csv("Diversity Output Text Level.csv", index= False, encoding='utf-8-sig')

#EMODIVERSITY - PERSON LEVEL

#Concatenate all texts per participant
participant_texts = final_dataset.groupby('alias')['replaced_lemmas'].apply(lambda x: ' '.join(x)).reset_index()

#Apply the emodiversity function to the concatenated texts
#Call the function for each text (filtered) to calculate diversity scores
person_diversity = participant_texts['replaced_lemmas'].apply(
    lambda text: get_emodiversity_scores(text, positive_emotion_lemmas, negative_emotion_lemmas)
)

person_diversity = pd.DataFrame(person_diversity.tolist())

person_diversity.to_csv(r"Diversity_person level.csv", index = False, encoding = 'utf-8-sig', sep =';')

#Load the files with liwc and mental mean values
liwc_text = pd.read_csv(r"LIWC_values.csv", encoding= 'utf-8-sig')
mental_mean = pd.read_csv(r"mental_}mean.csv", encoding = 'utf-8-sig')

#Only select the columns relevant for person level output - posemo, negemo, connectionId to group, alias
liwc_text = liwc_text[['connectionId', 'alias', 'posemo', 'negemo']]

# Group by connectionId and average posemo and negemo
liwc_person = liwc_text.groupby(['connectionId', 'alias']).agg(
    posemo_person_avg = ('posemo', 'mean'),
    negemo_person_avg = ('negemo', 'mean')
).reset_index()

# Keep only alias in person_diversity, drop from others before merging
person_combined = person_diversity.merge(
    liwc_person, 
    on='alias', 
    how='left'
) \
.merge(
    mental_mean,  
    on='alias', 
    how='left'
)

person_combined.to_csv(r"Diversity Output Person Level.csv",     
    index=False,
    sep=';',
    encoding='utf-8-sig')
