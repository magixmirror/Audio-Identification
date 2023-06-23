# Assignment in the course: Speech and Audio Processing
## Author: Ilias Tzanis

## Introduction
The purpose of this project is to develop a Covid-19 patient identification system using audio recordings. Specifically, it is based on the data used for the IEEE HealthCare Summit 2021 competition.

How to use it:
1. Clone the repository by running the following command in the command prompt: `git clone https://github.com/iliastzanis/Ergasia`.
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

# Εργασία στο μάθημα: Επεξεργασία Ομιλίας και ήχου
## Συγγραφέας: Ηλίας Τζανής

## Εισαγωγή 
Σκοπός της εργασίας αυτής είναι η κατασκευή ενός συστήματος αναγνώρισης ασθενών covid με την χρήση ηχητικών καταγραφών. Συγκεκριμένα στηρίχθηκα στα δεδομένα που χρησιμοποιήθηκαν για τον διαγωνισμό ΙΕΕΕ HealthCare Summit 2021. 

Πως να το χρησιμοποιήσεις;
1. Στο cmd τρέχουμε: git clone https://github.com/iliastzanis/Ergasia
2. Κατεβάζουμε τα δεδομένα του διαγωνισμού από την επίσημη σελίδα του.
3. Τρέχουμε: python create_dataset.py dataset_path path_images για την μετατροπή των ηχητικών καταγραφών σε μορφή εικόνων melspectogram. Οι εικόνες αυτές θα εποθηκευτούν στο path_images, ενώ στην μεταβήτή dataset_path βάζουμε το path των ηχητικών αρχείων που καταυάσαμε από το βήμα 2. 
4. Για να εκπαιδευσούμε τα μοντέλα τρέχουμε την εντολή: python train.py path_images length batch_size epochs steps_per_epoch trained_path, όπου:
    - path_images: είναι το path που αποθηκευσάμε τις εικόνες από το βήμα 3
    - length: είναι το μέγεθος των spectograms ειδόσου στο σύστημα
    - batch_size: το batch_size για την εκπαίδευση των δικτύων και εξαρτάται από την υπολογιστική ισχύ που διαθέτει το σύστημά μας
    - epochs: Ο αριθμός των εποχών εκπαίδευσης
    - steps_per_epoch: Ο αριθμός των batches ανά εποχή
    - trained_path: Ο φάκελος στον όποιον αποθηκεύονται τα βάρη των δικτύων μετά την εκπαίδευση
5. Για την αξίολογηση του συστήματος τρέχουμε την εντολή: python evaluate.py trained_path image_path με το path των pretrained weights που παρήχθησαν από το βήμα 4 και το path των εικόνων που παρήχθησαν από το βήμα 3.
