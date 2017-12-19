from transform import transformMEL
import sys, os 
import util
import numpy as np
import scipy 
import scipy.io.wavfile
import matplotlib 
import matplotlib.pyplot as plt

tr = transformMEL(bins=43, frameSize=1024, hopSize=512)

path_to_irmas = '/home/js/dataset/IRMAS/'
feature_dir_train = os.path.join(path_to_irmas,'features','Training')
if not os.path.exists(feature_dir_train):
    os.makedirs(feature_dir_train)

d=os.path.join(path_to_irmas,'Training')
instruments = sorted(filter(lambda x: os.path.isdir(os.path.join(d, x)), os.listdir(d)))

for count,inst in enumerate(instruments):
    for f in os.listdir(os.path.join(d,inst)):
        if os.path.isfile(os.path.join(d,inst, f)) and f.endswith('.wav'):
            audio, sampleRate, bitrate = util.readAudioScipy(os.path.join(d,inst,f)) 
            tr.compute_transform(audio.sum(axis=1),out_path=os.path.join(feature_dir_train,f.replace('.wav','.data')),suffix='_mel_',sampleRate=sampleRate)
            util.saveTensor(np.array([count],dtype=float),out_path=os.path.join(feature_dir_train,f.replace('.wav','.data')),suffix='_label_')

suffix_in='_mel_'
suffix_out='_label_'
file_list = [f for f in os.listdir(feature_dir_train) 
            if f.endswith(suffix_in+'.data') and 
            os.path.isfile(os.path.join(feature_dir_train,f.replace(suffix_in,suffix_out))) ]
print 'training file list: \n'+str(file_list)

melspec = util.loadTensor(out_path=os.path.join(feature_dir_train,file_list[0]))
# plt.imshow(np.log10(1+100*melspec.T),interpolation='none', origin='lower')
# plt.imshow(melspec.T,interpolation='none', origin='lower')

# plt.show()
# print 'input spectrogram shape '+str(melspec.shape)
# label = util.loadTensor(out_path=os.path.join(feature_dir_train,file_list[0].replace('mel','label')))
# print 'label of the instrument '+str(label)+', representing '+instruments[int(label)]