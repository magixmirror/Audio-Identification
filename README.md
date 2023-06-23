# Assignment in the course: Speech and Audio Processing
## Author: Ilias Tzanis

## Introduction
The purpose of this project is to develop a Covid-19 patient identification system using audio recordings. Specifically, it is based on the data used for the IEEE HealthCare Summit 2021 competition.

How to use it:
1. Clone the repository by running the following command in the command prompt: `git clone DataAnalysis-Covid-Audio-Identification`.
2. Download the competition data from the official website.
3. Run: `python create_dataset.py dataset_path path_images` to convert the audio recordings to mel spectrogram images. The generated images will be saved in the `path_images` directory, and in the `dataset_path` argument, provide the path to the audio files obtained from step 2.
4. To train the models, run the command: `python train.py path_images length batch_size epochs steps_per_epoch trained_path`, where:
   - `path_images` is the path where the images from step 3 are stored.
   - `length` is the size of the input spectrograms in the system.
   - `batch_size` is the batch size for network training, depending on the computational power available.
   - `epochs` is the number of training epochs.
   - `steps_per_epoch` is the number of batches per epoch.
   - `trained_path` is the folder where the network weights will be saved after training.
5. To evaluate the system, run the command: `python evaluate.py trained_path image_path` with the path to the pretrained weights generated from step 4, and the path to the images generated from step 3.

# Εργασία στο μάθημα: Επεξεργασία Ομιλίας και Ήχου
## Συγγραφέας: Ηλίας Τζανής

## Εισαγωγή 
Σκοπός της εργασίας αυτής είναι η κατασκευή ενός συστήματος αναγνώρισης ασθενών covid με την χρήση ηχητικών καταγραφών. Συγκεκριμένα στηρίχθηκα στα δεδομένα που χρησιμοποιήθηκαν για τον διαγωνισμό ΙΕΕΕ HealthCare Summit 2021. 

Πως να το χρησιμοποιήσεις;
1. Στο cmd τρέχουμε: `git clone DataAnalysis-Covid-Audio-Identification`.
2. Κατεβάζουμε τα δεδομένα του διαγωνισμού από την επίσημη σελίδα του.
3. Τρέχουμε: `python create_dataset.py dataset_path path_images` για τη μετατροπή των ηχητικών καταγραφών σε εικόνες melspectogram. Οι εικόνες θα αποθηκευτούν στο `path_images` φάκελο, ενώ στο `dataset_path` υποδεικνύουμε το path των ηχητικών αρχείων από το βήμα 2.
4. Για εκπαίδευση των μοντέλων, τρέχουμε την εντολή: `python train.py path_images length batch_size epochs steps_per_epoch trained_path`, όπου:
   - `path_images` είναι το path όπου αποθηκεύονται οι εικόνες από το βήμα 3.
   - `length` είναι το μέγεθος των εισόδων spectrograms στο σύστημα.
   - `batch_size` είναι το μέγεθος του batch για την εκπαίδευση του δικτύου, ανάλογα με τη διαθέσιμη υπολογιστική ισχύ.
   - `epochs` είναι ο αριθμός των εποχών εκπαίδευσης.
   - `steps_per_epoch` είναι ο αριθμός των batches ανά εποχή.
   - `trained_path` είναι ο φάκελος όπου θα αποθηκευτούν τα βάρη του δικτύου μετά την εκπαίδευση.
5. Για την αξιολόγηση του συστήματος, τρέχουμε την εντολή: `python evaluate.py trained_path image_path` με το path των προεκπαιδευμένων βαρών που δημιουργήθηκαν από το βήμα 4 και το path των εικόνων που δημιουργήθηκαν από το βήμα 3.
