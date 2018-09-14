# Music-classification-into-genres
The objective of this project is to analyse and classify a given music data into genre it belongs to; using Artificial Neural networks.

## Abstract
Neural networks have found profound success in the area of pattern recognition. By repeatedly showing a neural network inputs classified into groups, the network can be trained to discern the criteria used to classify, and it can do so in a generalized manner allowing successful classification of new inputs not used during training. <br>
Digital music is becoming more and more popular in people’s life. It is quite common for a person to own thousands of digital music pieces these days, and users may build their own music library through music management systems or software such as the Music Match Jukebox. However for professional music databases, labors are often hired to manually classify and index music assets according to predefined criteria; most users do not have the time or patience to browse through their personal music collections and manually index the music pieces one by one. On the other hand, if music assets are not properly classified, it may become a big headache when the user wants to search for a certain music piece among the thousands of pieces in a music collection. Manual classification methods cannot meet the development of digital music. Music classification is a pattern recognition problem which includes extraction features and establishing classifier.

## Approach
Neural network provides a new solution for music classification, so a new music classification method is proposed based on Back Propagation neural network in this experiment.<br>
<p>In this project, python framework is used for training the neural network that uses music songs data set. Data set contains features from symbolic songs (MP3, in this case) and uses them to classify the recordings by genre. Each example is classified as classic, rock, jazz or folk song. Further, there will be different data sets depending of features which will be taken. The attributes are duration of song, tempo, root mean square (RMS) amplitude, sampling frequency, sampling rate, dynamic range, tonality and number of digital errors.</p>
<p>Main goal of this experiment is to train neural network to classify this 4 type of genre and to discover which observed features has impact on classification. Data set contains 100 instances (25 of each genre), 8 numeric attributes and genre name. Each instance has one of 4 possible classes: classic, rock, jazz or folk.</p>

## Dataset description
Data set contains features from symbolic songs. Data set contains 100 instances (25 of each genre), 8 numeric attributes and genre name. Each instance has one of 4 possible classes: classic, rock, jazz or folk.<br>
<b>Attribute Information:</b>
1. Song's duration in seconds
2. Tempo in beats per minute (BPM)
3. Root mean square (RMS) amplitude in dB. The RMS (Root-Mean-Square) value is the effective value of the total waveform
4. Sampling frequency in kHz
5. Sampling rate in b.
6. Dynamic range(dr) in dB. Dynamic range, is the ratio between the largest and smallest possible values of a changeable quantity, such as in signals like sound and light.
7. Tonality can be C, C#, D, D#, E, F, F#, G, G#, A, Bb and B, with associated values from 0 to 11 respectively.
8. Number of digital errors(nde) - There are two types of digital errors, glitches and clipping
9. Genre name: classic, rock, jazz and folk

## Data Preprocessing
Normalising the data: Normalization implies that all values from the data set should take values in the range from 0 to 1. The purpose of this stage is to reduce the audio files down to a low dimensional representation which a practically-sized network can then take in as inputs.<br>
Technique used- Min-max normalisation

## Implementation
<b>Model used:</b> Back Propagation
<b>Learning rate:</b> 0.05
<b>Framework used:</b> Python



