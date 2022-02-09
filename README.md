Final Project for EE341 

In the past quarter (Fall 2021) I have worked with the mlcommons group to help develop and autoencoder to detect anomalies in audio signals.
 For a brief background of use cases for auto-encoders such as this one, automation and
AI based algorithms provide great flexibilities for industrial systems to achieve higher
productivity, resource efficiency and reduced lead times. Detection of failures or anomalous
behavior within these systems is critical to operation and monitoring of audio emitted from
machines may be useful for machine operation state tracking. Conditions in industrial factories
do not allow for every possible type of anomaly so this makes simple binary classification
impractical. To achieve tracking, the model must identify unknown (out of sample) sounds while
only training on the normal audio samples.
 The raw data therefore comes in the form of audio samples, which are labeled based on
the machine type and specific machine that produced them. For each machine, each of the four
machine IDs has over 1000 training samples and around 615 test audio samples.
 The audio files are then processed such that each audio file has 200 overlapping STFTs
that summarize frequency signature in 128 distinct bands to create a mel spectrogram. 

The scope for my project would then be to experiment with the different parameters for the data
pre-processing. This includes: 

* Frames
* Hop length (number of samples between successive frames)
* N_fft (length of the FFT window)
* N_mels (number of mel bins)
* Power (Exponent for the magnitude melspectrogram) 

By the end of the project, I have shown how each of these parameters directly affect the
training data and showcase the effects of these different parameters to model performance. 
