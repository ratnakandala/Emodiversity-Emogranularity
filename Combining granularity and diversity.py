import csv
import pandas as pd

#For all columns - Dataset with granularity values for texts filtered for length < 15
revised_granularity = pd.read_csv(r"Diversity Output Text Level.csv", encoding='utf-8-sig')

#For all columns
revised_diversity = pd.read_csv(r"Granularity Output Text Level.csv", encoding='utf-8-sig')

#First merge granularity and the diversity datasets as they are of the same length and based on the same filtered dataset
merged_grandiv= revised_granularity.merge(revised_diversity, 
                                        left_index=True,
                                        right_index=True
                                        )

merged_grandiv.to_csv("Input for MI.csv", index = False, encoding='utf-8-sig')                                        