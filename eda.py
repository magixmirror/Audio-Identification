# αρχικά εισάγουμε τις απαραίτητες βιβλιοθήκες 
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import soundfile as sf
import matplotlib

from tqdm import tqdm 
from PIL import Image
from librosa import power_to_db,amplitude_to_db,stft
from librosa.feature import melspectrogram



# ορισμός του path που βρίσκονται τα αρχεία με τις ηχογραφήσεις
path = "IEEE_HealthCareSummit_Dev_Data_Release"


# αρχικά βλέπουμε τι είδους αρχεία έχει ο φάκελος με τις ηχογραφήσεις 
print (f"Files in data: {os.listdir(path)}")

print (f"Subfolders of AUDIO: {os.listdir(os.path.join(path, 'AUDIO'))}")


metadata = pd.read_csv(os.path.join(path, "metadata.csv"), sep=" ") # διαβάζουμε το αρχείο των μεταδεδομένων και 
print(metadata.head()) # τυπώνουμε το head ώστε να δούμε πως είναι δομημένα τα δεδομένα 

# αρχικά βλέπουμε την κατανομή των φύλλων στα άτομα 
gender_hist = {"m": 0, "f": 0} # αρχικοποιούμε 2 μετρητές ανα φύλλο με 0
for gender in metadata["GENDER"]: # για κάθε πεδίο της στήλης gender
  gender_hist[gender] += 1 # πάμε στο αντίστοιχο φύλλο και προσθέτουμε +1 στην τιμή


plt.bar(list(gender_hist.keys()), list (gender_hist.values())) # τυπώνουμε το διάγραμμα
plt.show()

print (f"Num of males / num of females = {gender_hist['m']/ gender_hist['f'] * 100}%") # ποσοστο των male/female

# στην συνέχεια θα μελετήσουμε την κατανομή των covid status στο σύνολο δεδομένων

covid_hist = {"n": 0, "p": 0} # αρχικοποιούμε 2 μετρητές ανα φύλλο με 0
for covid in metadata["COVID_STATUS"]: # για κάθε πεδίο της στήλης gender
  covid_hist[covid] += 1 # πάμε στο αντίστοιχο φύλλο και προσθέτουμε +1 στην τιμή

plt.bar(list(covid_hist.keys()), list (covid_hist.values())) # τυπώνουμε το διάγραμμα
plt.show()

print (f"Num of positives / num of negatives = {covid_hist['p']/ covid_hist['n'] * 100}%")

# επιλέγουμε ένα τυχαίο άτομο (τυχαίο id) και τυπώνουμε σε διαγραμμα τα ηχητικά του αρχεία 
id = np.random.choice (metadata["SUB_ID"])
breathing,sr = sf.read(os.path.join(path,"AUDIO","breathing", f"{id}.flac"))
plt.subplot(3, 1, 1)
plt.plot(breathing)
plt.title("Breathing")

# το ίδιο για την ομιλία 
speech,sr = sf.read(os.path.join(path,"AUDIO","speech", f"{id}.flac"))
plt.subplot(3, 1, 2)
plt.plot(speech)
plt.title("Speech")


# και το ίδιο και για τον βήχα 
cough,sr = sf.read(os.path.join(path,"AUDIO","cough", f"{id}.flac"))
plt.subplot(3, 1, 3)
plt.plot(cough)
plt.title("cough")

plt.show()


breathing_stft = stft(breathing) # παίρνουμε το stft ενός σήματος 
breathing_stft_db = amplitude_to_db(breathing_stft,ref=np.max) # το μετατρέπουμε σε μορφή db
breathing_stft_db = breathing_stft_db[:128] # και επιλέγουμε μόνο τις 128 πρώτες συχνότητες 

plt.subplot(2, 1, 1)
plt.plot (breathing) # τέλος τυπώνουμε το αρχικό διάγραμμα
plt.title("breathing")

plt.subplot(2, 1, 2)
plt.imshow(breathing_stft_db) # και το συχνοτικό διάγραμμα
plt.title("breathing stft")
plt.show()

# επαναλαμβάνουμε την διαδικασία αυτή και για τα άλλα 2 σήματα 

cough_stft = stft(cough) # παίρνουμε το stft ενός σύματος 
cough_stft_db = amplitude_to_db(cough_stft,ref=np.max) # το μετατρέπουμε σε μορφή db
cough_stft_db = cough_stft_db[:128] # και επιλέγουμε μόνο τις 128 πρώτες συχνότητες 

plt.subplot(2, 1, 1)
plt.plot (cough) # τέλος τυπώνουμε το αρχικό διάγραμμα
plt.title("cough")

plt.subplot(2, 1, 2)
plt.imshow(cough_stft_db) # και το συχνοτικό διάγραμμα
plt.title("cough stft")
plt.show()

# τέλος και για το σήμα της ομιλίας

speech_stft = stft(speech) # παίρνουμε το stft ενός σύματος 
speech_stft_db = amplitude_to_db(speech_stft,ref=np.max) # το μετατρέπουμε σε μορφή db
speech_stft_db = speech_stft_db[:128] # και επιλέγουμε μόνο τις 128 πρώτες συχνότητες 

plt.subplot(2, 1, 1)
plt.plot (speech) # τέλος τυπώνουμε το αρχικό διάγραμμα 
plt.title("speech")

plt.subplot(2, 1, 2)
plt.imshow(speech_stft_db) # και το συχνοτικό διάγραμμα
plt.title("speech stft")
plt.show()


# δοκιμή melspectrogram

breathing_mel = melspectrogram(breathing,sr) 
breathing_mel_db = power_to_db(breathing_mel, ref=np.max)


plt.subplot(2, 1, 1)
plt.plot (breathing) # τυπώνουμε το αρχικό διάγραμμα και το συχνοτικό δίαγραμμα που εγινε με τη χρηση της melspectrogram
plt.title("breathing")

plt.subplot(2, 1, 2)
plt.imshow (breathing_mel_db)
plt.title("breathing melspectrogram")
plt.show()

plt.subplot(2, 1, 1)
plt.imshow(breathing_stft_db) # τυπώνουμε πρώτα το stft και ύστερα το melspectrogram διάγραμμα
plt.title("breathing stft")

plt.subplot(2, 1, 2)
plt.imshow(breathing_mel_db)
plt.title("breathing melspectrogram")

plt.show()