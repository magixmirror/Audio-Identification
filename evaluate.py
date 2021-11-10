from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import pandas as pd
import argparse
import os
from tqdm import tqdm 
import numpy as np
from PIL import Image



# Για να διαβάσουμε τα ορισμάτα εισόδου από το cmd
parser = argparse.ArgumentParser()
parser.add_argument("trained_path", help="Path where the models weights after training will be stored", default = "pretrained-weights")
parser.add_argument("image_path", help="Path to dataset", default = "images")
args = parser.parse_args()
trained_path = args.trained_path
image_path = args.image_path

# load pretrained weights 
breathing_model = load_model(os.path.join(trained_path, "breathing.h5"))
cough_model = load_model(os.path.join(trained_path, "cough.h5"))
speech_model = load_model(os.path.join(trained_path, "speech.h5"))

# Test models Accuracy 
def predict_subject_system(sub_id, system, length = 128):
  predictions = [] # αρχικοποιούμε τα predictions 
  # read image 
  im = np.array(Image.open(os.path.join(image_path, system, f"{sub_id}.png")))
  inp = []
  if (im.shape[1] < length):
    return []

  for index in range(im.shape[1] - length):
    inp.append(im[:, index : index + length])
  
  inp = np.array(inp).reshape((-1, 128, length, 1))
  if system == "breathing":
    pred = breathing_model.predict(inp)
  elif system == "cough":
    pred = cough_model.predict(inp)
  elif system == "speech":
    pred = speech_model.predict(inp)
  else:
    print (f"{system} is not a valid system!")
    return None 

  predictions += [p[0] for p in pred]
  return predictions 


def predict_subject(sub_id, length = 128):
  breath_pred = predict_subject_system(sub_id, "breathing", length)
  cough_pred = predict_subject_system(sub_id, "cough", length)
  speech_pred = predict_subject_system(sub_id, "speech", length)
  
  sub_pred = breath_pred + cough_pred + speech_pred
  return np.mean(sub_pred)


test_rows = pd.read_csv(os.path.join(image_path, "test_metadata.csv")) # και test γραμμές 

# test the system
true_labels = []
predictions = []
for row in tqdm(test_rows.iterrows()):
  sub_id = row[1]["SUB_ID"]
  covid_status = row[1]["COVID_STATUS"]

  # get true labels
  if covid_status == "n":
    true_labels.append(0)
  else:
    true_labels.append(1)
  # get predictions 
  pred = predict_subject(sub_id)
  if pred >= 0.5:
    predictions.append(1)
  else:
    predictions.append(0)


# αφού κατασκευάσουμε τα predictions και τις πραγματικές τιμές θα τεστάρουμε το σύστημά μας χρησιμοποιόντας τις 3
# πιο γνωστές τενχικές για το task αυτό accuracy, precision, f1
acc = accuracy_score(true_labels[:len(predictions)], predictions)
prec = precision_score(true_labels[:len(predictions)], predictions)
f1 = f1_score(true_labels[:len(predictions)], predictions)


print (f"Accuracy: {acc}, Precision: {prec}, F1: {f1}")

