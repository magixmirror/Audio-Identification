# Σχεδιασμός του μοντέλου αναγνώρισης των ασθενών 
from tensorflow.keras.layers import Dense,Flatten, Input, Conv2D,MaxPool2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pandas as pd # για το δίαβασμα των csv αρχείων 
import os
import argparse
import numpy as np
from PIL import Image 


# Για να διαβάσουμε τα ορισμάτα εισόδου από το cmd
# ορίζουμε τις μεταβλητές που θέλουμε να διαβάοσυμε 
parser = argparse.ArgumentParser()
parser.add_argument("image_path", help="Path to dataset", default = "images")
parser.add_argument("length", help="Sequense length", default = 128)
parser.add_argument("batch_size", help="Training batch size", default = 64)
parser.add_argument("epochs", help="Number of training epochs", default = 10)
parser.add_argument("steps_per_epoch", help="Steps or batches per epoch", default = 1000)
parser.add_argument("trained_path", help="Path where the models weights after training will be stored", default = "pretrained-weights")
# διαβάζουμε τις μεταβλητές από την είσοδο
args = parser.parse_args()
# και τις αποθηεκυούμε σε δικές μας μεταβλητές για να τις χρησιμοποιούμε ευκολότερα 
image_path = args.image_path
length = int (args.length)
batch_size = int (args.batch_size)
epochs = int (args.epochs)
steps_per_epoch = int (args.steps_per_epoch)
trained_path = args.trained_path


# η συνάρτηση κατασκευής των νευρωνικών δικτύων 
def create_model(system, length = 256):
    inp = Input((128,length,1)) # η διάσταση εισόδου
    outp = Conv2D(32,16,activation='relu')(inp) # το στρώμα CONV
    outp = MaxPool2D(2)(outp) # το στρώμα max-pooling
    outp = Dropout(0.2)(outp) # το στρώμα dropout για να λύσουμε το πρόβλημα της ανισοκατανομής των κλάσεων 
    outp = Conv2D(64,16,activation='relu')(outp)
    outp = MaxPool2D(2)(outp)
    outp = Dropout(0.2)(outp)
    outp = Conv2D(64,16,activation='relu')(outp)
    outp = Flatten()(outp)
    outp = Dropout(0.2)(outp)
    outp = Dense(1,activation='sigmoid')(outp) # κάνουμε όλες τις εξόδους σε μια διάσταση από 0 σε 1 λόγω της sigmoid 
    m = Model(inp, outp, name=system) # οριζούμε το μοντέλο και τον optimizer και το loss function μαζί με τις μετρικές που θα χρησιμοποιήσουμε που είναι η accuracy
    m.compile(optimizer = Adam(lr=10e-5), loss='binary_crossentropy', metrics=['acc'])
    return m


# generator για κατακσευή των δεδομένων για την εκπαίδευση του δικτύου
def train_generator(system, length = 256, batch_size = 128):  
  
  positive_ids = train_rows[train_rows['COVID_STATUS'] == 'p']["SUB_ID"].tolist() # βρίσκουμε τα θετικά δείγματα
  negative_ids = train_rows[train_rows['COVID_STATUS'] == 'n']["SUB_ID"].tolist() # και τα αρνητικά δείγματα
  while True:
    # select random samples 
    # equal number for covid and non-covid subjects
    positive_samples = np.random.choice(positive_ids, batch_size // 2) # επιλέγουμε τυχαία τα μισά δείγματα ως θετικά 
    negative_samples = np.random.choice(negative_ids, batch_size // 2) # και τα άλλα μισά ως αρνητικά για να εχουμε ισαμοιρασμένο batch 

    x_batch, y_batch = [], []
    # iterate over positive samples and create the batch 
    for p_id in positive_samples: # για όλα τα θετικά δείγματα 
      im = np.array(Image.open(os.path.join(image_path, system, f"{p_id}.png"))) / 255.0 # φορτώνουμε την εικόνα και την κάνουμε από 0 σε 1
      if im.shape[1] > 2 * length: # σπάμε την εικόνα σε τυχαία παράθυρα μεγέθους length
        starting_point = np.random.choice(range (im.shape[1] - length - 1)) # select random starting point 
        im_slice = im[:, starting_point : starting_point + length]

        x_batch.append(im_slice)
        y_batch.append(1)

    # same for negative samples
    for n_id in negative_samples:
      im = np.array(Image.open(os.path.join(image_path, system, f"{n_id}.png"))) / 255.0
      if im.shape[1] > 2 * length:
        starting_point = np.random.choice(range (im.shape[1] - length - 1)) # select random starting point 
        im_slice = im[:, starting_point : starting_point + length]

        x_batch.append(im_slice)
        y_batch.append(0)

    yield np.array(x_batch), np.array(y_batch)



train_rows = pd.read_csv(os.path.join(image_path, "train_metadata.csv")) # φορτώνουμε τις train 
test_rows = pd.read_csv(os.path.join(image_path, "test_metadata.csv")) # και test γραμμές 


# Κατασκευάζουμε τα 3 μοντέλα, ένα για κάθε σύστημα
breathing_model = create_model("breathing", length)
speech_model = create_model("speech", length)
cough_model = create_model("cough", length)


# Εκπαιδευούμε κάθε ένα μοντέλο 
# αρχικά κατασκευζουμε τους 3 generators
breathing_gen = train_generator("breathing", length, batch_size = batch_size)
cough_gen = train_generator("cough", length, batch_size = batch_size)
speech_gen = train_generator("speech", length, batch_size = batch_size)
# και στην συνέχεια με βάση αυτούς τα εκπαιδευούμε 
breathing_model.fit_generator(breathing_gen, steps_per_epoch = steps_per_epoch, epochs=epochs)
cough_model.fit_generator(cough_gen, steps_per_epoch = steps_per_epoch, epochs=epochs)
speech_model.fit_generator(speech_gen, steps_per_epoch = steps_per_epoch, epochs=epochs)


# αφού τελειώσουμε την εκπαίδευση σώζουμε τα βάρη του δικτύου
if not os.path.exists(trained_path): # αν ο φάκελος για τα βάρη δεν υπάρχει
    os.makedirs(trained_path) # τον κατασκευάζουμε 

# σώζουμε τα βάρη των εκπαιδευμένων δικτύων 
breathing_model.save(os.path.join(trained_path, "breathing.h5"))
cough_model.save(os.path.join(trained_path, "cough.h5"))
speech_model.save(os.path.join(trained_path, "speech.h5"))
