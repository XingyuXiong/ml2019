import numpy as np
from scipy.io import wavfile
import matplotlib.pylab as plt


sampling_freq,audio=wavfile.read('1.wav')

print('\nShape:',audio.shape)
print('Datatype:',audio.dtype)
print('Duration:',round(audio.shape[0]/float(sampling_freq),3),'seconds')
print(audio[:10000:1000])
#audio=audio/(2.**15)
audio=audio[:]

'''
x_values=np.arange(0,len(audio),1)/float(sampling_freq)
plt.plot(x_values,audio,color='black')
plt.xlabel('Time(ms)')
plt.ylabel('Amplitude')
plt.title('Audio signal')
plt.show()
'''

pca_wav=wavfile.write('2.wav',sampling_freq,audio)