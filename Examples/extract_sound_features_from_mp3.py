
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import librosa
import numpy as np
import os, glob, sys
import matplotlib as mpl
mpl.rc('font',family='Times New Roman')
from pylab import *
import random


def dumpToBinaryGenre(dataFolder='/a/joe_data/MIR_Dataset/Music_Audio_Benchmark_Data_Set/'): 
    genres = []
    for root, dirs, files in os.walk(dataFolder):
        if not dirs:
            genres+=[root]
    print genres
        
    
    
def dumpToBinary(binFileName='/a/joe_data/MIR_Dataset/Music_mood_dataset/set1/mp3/Soundtrack360_mp3/'):
  
  # get labels of music
  labelFile = '/a/joe_data/MIR_Dataset/Music_mood_dataset/set1/mp3/Soundtrack360_mp3/song_mood_label.txt'
  myFile = open(labelFile)
  myLabels = myFile.readlines()
  myLabels = [ _ptr.rstrip() for _ptr in myLabels]
  moodSets = list(set(myLabels))
  moodSets.sort()

  
  musicMoodToNumeric = dict(zip( moodSets, xrange(len(moodSets)) ))
  musicNumericToMood = dict(zip( xrange(len(moodSets)), moodSets) )
  
  print "Here is the list of moods %s with size of %s " % ( moodSets, len(moodSets))

  myLabels = np.array([ musicMoodToNumeric[_ptr]  for _ptr in myLabels ]).astype(np.uint8)

  # get audio-image of music
  # chromagram
  data_folder = '/a/joe_data/MIR_Dataset/Music_mood_dataset/set1/mp3/Soundtrack360_mp3/chromagram/'
  # tempogram
  #data_folder = '/a/joe_data/MIR_Dataset/Music_mood_dataset/set1/mp3/Soundtrack360_mp3/tempogram/'
  
  all_files = []
  for root, dirs, files in os.walk(data_folder):
    files = glob.glob(os.path.join(root, '*.jpg'))
    all_files += files
  all_files.sort() # note here all the file names are like 000.jpg, 001.jpg ...., so sorted will align it with label list

  IMAGE_Y_LENGTH = 50
  IMAGE_X_LENGTH = 50
  SIZE_PER_IMAGE = 1

  with open(binFileName,'wb') as myFile, open(binFileNameTest, 'wb') as myFileTest:
    for index, fileName in enumerate(all_files):

      labelMood = musicNumericToMood[ myLabels[index] ]
      
      # this is for loading numpy .npy files
      #audioImage = np.load(fileName).astype(np.uint8)

      # this is for loading jpg image files
      audioImage = np.array(Image.open(fileName)).astype(np.uint8)
                  
      TOTAL_X_LENGTH = audioImage.shape[1]
      TOTAL_Y_LENGTH = audioImage.shape[0]

      # this is for bootstrapping mode samples from one image
      for itr in xrange(SIZE_PER_IMAGE):
        # we can choose to crop some random portion of the images
        startX = random.randint(0, TOTAL_X_LENGTH - IMAGE_X_LENGTH  )
        startY = random.randint(0, TOTAL_Y_LENGTH - IMAGE_Y_LENGTH  )
        croppedImage = audioImage[startY:startY+IMAGE_Y_LENGTH, startX:startX+IMAGE_X_LENGTH]

        # switch axises now, for R -> G -> B
        croppedImage = croppedImage.transpose([2,0,1])

        redImage = croppedImage.flatten()

        # this below is needed for loading numpy files
        #greenImage = np.zeros(len(redImage)).astype(np.uint8)
        #blueImage = np.zeros(len(redImage)).astype(np.uint8)


        # test file
        if labelMood in moodSets:
          print 'Ingest to test file!', labelMood, moodSets
          myLabels[index].tofile(myFileTest)
          redImage.tofile(myFileTest)
          moodSets.remove(labelMood)

        else:
          # train file
          myLabels[index].tofile(myFile)
          redImage.tofile(myFile)

       
        
          # this is for numpy file loading
          #greenImage.tofile(myFile)
          #blueImage.tofile(myFile)
        
      


def readFromBinary():
    # read cifar images
    #fp = open('/tmp/cifar10_data/cifar-10-batches-bin/data_batch_1.bin','rb')

    # read music images
    fp = open('/a/joe_data/MIR_Dataset/Music_mood_dataset/set1/mp3/Soundtrack360_mp3/batch_test.bin','rb')
  
    b = np.fromfile(fp, dtype=np.uint8)
    for itr in xrange(0,12,1):
      # this for cifar images
      #im = b[1+3073*itr:1+3073*itr+3072] # 1-> label, 3072=32*32*3,

      # this is for audio images
      im = b[1+7501*itr:1+7501*itr+7500] # 1-> label, 3072=32*32*3,

      # cifar images
      #im1 = im.reshape((3,32,32)) # red, green, blue

      # audio images
      im1 = im.reshape((3,50,50)) # red, green, blue

      im2 = im1.transpose([1,2,0])
      import matplotlib.pyplot as plt
      imgplot = plt.imshow(im2)
      plt.show()

def convertMP3ToNumpyArray():
    data_folder = '/a/joe_data/MIR_Dataset/Music_mood_dataset/set1/mp3/Soundtrack360_mp3/'
    all_files = []
    for root, dirs, files in os.walk(data_folder):
        files = glob.glob(os.path.join(root, '*.mp3'))
        all_files += files
    
    for fileName in all_files:
        y, sr = librosa.load(fileName)
        C = librosa.feature.chroma_cqt(y=y, sr=sr)
        oenv = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.estimate_tempo(oenv, sr=sr)
        Tgram = librosa.feature.tempogram(y=y, sr=sr)
        Tgram = Tgram[0:100]
        # MEL
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)
        mel_spectrogram =  librosa.logamplitude(S,ref_power=np.max)
        

        figure(figsize = (5,5)) # 5 x 5 inch
        ax = Axes(plt.gcf(),[0,0,1,1],yticks=[],xticks=[],frame_on=False)
        plt.gcf().delaxes(plt.gca())
        plt.gcf().add_axes(ax)

        # MEL
        #plt.imshow(mel_spectrogram, aspect='auto')
        # TEMPO GRAM
        #plt.imshow(Tgram, aspect='auto')
        # CHROMAGRAM
        plt.imshow(C, aspect='auto')
        
        #plt.imshow(mel_spectrogram)
        #plt.show()
        
        songIndex= os.path.basename(fileName).split('.')[0]

        # this is for saving .npy files
        #np.save( data_folder+songIndex+'.npy', mel_spectrogram)

        # this is for saving .jpg files
        #plt.savefig(data_folder+'tempogram/'+songIndex+'.jpg', dpi=10) # 10 pixels per inch
        plt.savefig(data_folder+'chromagram/'+songIndex+'.jpg', dpi=10) # 10 pixels per inch
        plt.close('all')
        
  

def plotMusic():    
  # default sampling rate = 22050 (22 kHz)
  # return of y = audio time series data
  
  #y, sr = librosa.load('/a/joe_data/ximalaya_data/complete_database/wav_files/16197447.mp3')
  #y, sr = librosa.load('/a/joe_data/ximalaya_data/complete_database/wav_files/139679.mp3')

  y, sr = librosa.load('/a/joe_data/MIR_Dataset/Music_mood_dataset/set1/mp3/Soundtrack360_mp3/091.mp3')


  print y.shape, ' Sampling rate: %s Hz' % sr


  plt.figure(figsize=(16, 14), facecolor='white' )

  # in frequency domain, the signal amplitude square
  # compute db relative to peak power

  # power spectrum
  D = librosa.logamplitude(np.abs(librosa.stft(y))**2, ref_power=np.max)
  print 'Power spectrum (FFT), ', D.shape

  plt.subplot(4, 2, 1)
  librosa.display.specshow(D, y_axis='linear')
  plt.colorbar(format='%+2.0f dB')
  plt.title('Linear-frequency power spectrogram')


  # Or on a logarithmic scale, also power spectrum
  plt.subplot(4, 2, 2)
  librosa.display.specshow(D, y_axis='log')
  plt.colorbar(format='%+2.0f dB')
  plt.title('Log(frequency) power spectrogram')
  
  
  # Or use a CQT scale
  CQT = librosa.logamplitude(librosa.cqt(y, sr=sr)**2, ref_power=np.max)
  print 'Power spectrum (CQT)', CQT.shape
  
  plt.subplot(4, 2, 3)
  librosa.display.specshow(CQT, y_axis='cqt_note')
  plt.colorbar(format='%+2.0f dB')
  plt.title('Constant-Q power spectrogram (note)')

  plt.subplot(4, 2, 4)
  librosa.display.specshow(CQT, y_axis='cqt_hz')
  plt.colorbar(format='%+2.0f dB')
  plt.title('Constant-Q power spectrogram (Hz)')


  # Draw a chromagram with pitch classes, 12 pitch classes

  C = librosa.feature.chroma_cqt(y=y, sr=sr)
  print 'Chromagram ', C.shape

  plt.subplot(4, 2, 5)
  librosa.display.specshow(C, y_axis='chroma')
  plt.colorbar()
  plt.title('Chromagram')
  #plt.xlabel('Time')


  # Draw a tempogram with BPM markers

  plt.subplot(4, 2, 6)
  oenv = librosa.onset.onset_strength(y=y, sr=sr)
  print 'Spectral flux onset strength, envolope ', oenv.shape
  
  tempo = librosa.beat.estimate_tempo(oenv, sr=sr)
  print 'Tempo (BPM)  ', tempo

  # local autocorrelation of the onset strength envelope
  Tgram = librosa.feature.tempogram(y=y, sr=sr)
  print 'Tempo-gram ', Tgram.shape

  librosa.display.specshow(Tgram[:100], x_axis='time', y_axis='tempo',
                           tmin=tempo/4, tmax=tempo*2, n_yticks=4)


  plt.colorbar()
  plt.title('Tempogram')


  # now the mel-spectrolgram
  # 128 mel components/bands, hop size = 512, FFT window size 2048, f_max highest frequency 8000 Hz
  S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)
  mel_spectrogram =  librosa.logamplitude(S,ref_power=np.max)
  print 'mel-spectrogram', S.shape, mel_spectrogram.shape, S, mel_spectrogram

  plt.subplot(4, 2, 7)
  librosa.display.specshow(mel_spectrogram, y_axis='mel', fmax=8000, x_axis='time')
  plt.colorbar(format='%+2.0f dB')
  plt.title('MEL-spectrogram')


  plt.tight_layout()
  plt.show()



if __name__ == '__main__':
    #convertMP3ToNumpyArray()
    #readFromBinary()
    #dumpToBinary()
    dumpToBinaryGenre()
    
    


