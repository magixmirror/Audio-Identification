# αρχικά εισάγουμε τις απαραίτητες βιβλιοθήκες 
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import soundfile as sf
from tqdm import tqdm 
from PIL import Image
import argparse
from librosa.feature import melspectrogram
from librosa import power_to_db

# Για να διαβάσουμε τα ορισμάτα εισόδου από το cmd
parser = argparse.ArgumentParser()
parser.add_argument("path", help="Path to dataset", default = "IEEE_HealthCareSummit_Dev_Data_Release")
parser.add_argument("image_path", help="Path to dataset", default = "images")

args = parser.parse_args()

path = args.path
image_path = args.image_path


# χωρίζουμε τα δεδομένα σε train και test κατά την διαδικασία κατασκευής του dataset 
# οπότε σπάμε τους ασθενείς σε train και test σετ ώστε να μπορέσουμε να εκπαιδευσούμε
# αλλά και να αξιολογήσουμε τα αποτελέσματα του αλγορίθμου μας
# αρχικά διαβάζουμε τα metadata
metadata = pd.read_csv(os.path.join(path, "metadata.csv"), sep=" ")
# και στην συνέχεια σπαμε σε train και test
np.random.seed(42) # για να έχουμε πάντα την ίδια παραγωγή train και test μεταξύ πειραμάτων ακόμα και αν φορτώσουμε τα pretrained weights να μην αλλάξουν τα αποτελέσματα της αξιολόγησης 
ids = metadata["SUB_ID"].tolist()
test_ids = np.random.choice(ids, 200)
train_ids = [id for id in ids if id not in test_ids]

# αφού σπάσουμε τα ids σε train και test στην συνέχεια θα μελετήσουμε την κατανομή των κλάσεων 
# των ασθενών σε κάθε περίπτωση ώστε να δούμε κατά πόσο τα σύνο αυτά αντικατοπτρίζουν την πραγματική κατονομή των δεδομένων
train_rows = metadata[metadata["SUB_ID"].isin(train_ids)]
pos = len(train_rows[train_rows['COVID_STATUS'] == 'p'])
neg = len(train_rows[train_rows['COVID_STATUS'] == 'n'])
print (f"Print number of positives covid in train set: {pos}")
print (f"Print number of negatives covid in train set: {neg}")
print (f"Positive to negative Ratio on train set= {pos/neg * 100} %")
print ("-----------------------------------------------------------------------------------------")

# το ίδιο και για το test set
test_rows = metadata[metadata["SUB_ID"].isin(test_ids)]
pos = len(test_rows[test_rows['COVID_STATUS'] == 'p'])
neg = len(test_rows[test_rows['COVID_STATUS'] == 'n'])
print (f"Print number of positives covid in train set: {pos}")
print (f"Print number of negatives covid in train set: {neg}")
print (f"Positive to negative Ratio on test set= {pos/neg * 100} %")


# στην συνέχεια πάμε και εξάγουμε τις εικόνες από τα ηχητικά αρχεία
# τις εικόνες τις σώζουμε στο path που όρισε ο χρήστης 
if not os.path.exists(image_path): # αν ο φάκελος για τις εικόνες δεν υπάρχει
    os.makedirs(image_path) # τον κατασκευάζουμε 


# τέλος σώζουμε τα datasets τα train και test rows
test_rows.to_csv(os.path.join(image_path, "test_metadata.csv"))
train_rows.to_csv(os.path.join(image_path, "train_metadata.csv"))


for t in ['breathing','cough','speech']: # για κάθε ένα από τα 3 σήματα
  if not os.path.exists(os.path.join(image_path, t)): # αν ο υπο-φάκελος δεν υπάρχει
    os.makedirs(os.path.join(image_path, t)) # τον κατασκευάζουμε 
     
  for f in tqdm(os.listdir(path+'/AUDIO/'+t)): # διαβάζουμε όλα τα αρχεία για τον αντίστοιχο φάκελο
    audio, sr = sf.read(os.path.join(path,'AUDIO',t,f)) # διβάζουμε κάθε ένα ηχητικό αρχείο
    mel = melspectrogram(audio,sr) # κατασκευάζουμε το melspectogram του ηχητικού
    mel_db = power_to_db(mel, ref=np.max) 

    mel_db = (mel_db+80)/80*255 # το μετατρέπουμε σε μορφή εικόνας

    im = Image.fromarray(mel_db).convert('L') # κάνουμε τον πίνακα εικόνα
    im.save(os.path.join(image_path, t, f[:-5]+'.png')) # και τελος το αποθηκευούμε την εικόνα


