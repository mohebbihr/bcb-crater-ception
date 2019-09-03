# This script plot the number of craters based on radius size.
# It could be good to have it for non-crater too which I need to change the extract_samples script
# to save the extracted non-crater samples on a csv file. 
import pandas as pd


gt_csv_path = 'crater_data/gt/1_24_gt.csv'
gt = pd.read_csv(gt_csv_path, header=None)

