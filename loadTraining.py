from transform import transformMEL
import sys, os 
import util
import numpy as np
import scipy 
import scipy.io.wavfile
import matplotlib 
import matplotlib.pyplot as plt
import dataset
from dataset import MyDataset
path_to_irmas = '/home/js/dataset/IRMAS/'
feature_dir_train = os.path.join(path_to_irmas,'features','Training')
if not os.path.exists(feature_dir_train):
    print feature_dir_train + ' does not exist!'
    exit()

d=os.path.join(path_to_irmas,'Training')
instruments = sorted(filter(lambda x: os.path.isdir(os.path.join(d, x)), os.listdir(d)))

db=MyDataset(feature_dir=feature_dir_train, batch_size=128, time_context=128, step=50, 
             suffix_in='_mel_',suffix_out='_label_',floatX=np.float32,train_percent=0.8)
print "total number of instances: "+str(db.total_points)
print "batch_size: "+str(db.batch_size)
print "iteration size: "+str(db.iteration_size)
print "feature shape: "+str(db.features.shape)
feature_reshape = db.features.reshape(-1, 5504)
print "feature shape: "+str(feature_reshape.shape)
print "labels shape: "+str(db.labels.shape)
print "feature validation shape: "+str(db.features_valid.shape)
print "labels validation shape: "+str(db.labels_valid.shape)

features,labels = db()
#we did  np.swapaxes for tensorflow in self.iterate() so we do it backwards
features = np.swapaxes(features,1,3)

print "iteration step "+str(db.iteration_step)
print features.shape
print labels.shape
#we go back from categorical to numerical labels
label = np.nonzero(labels[2,:])[0]
print 'instrument: '+instruments[int(label)]

# plt.imshow(np.log10(1+100*features[2,0,:,:].T),interpolation='none', origin='lower')
plt.imshow(features[2,0,:,:].T,interpolation='none', origin='lower')
plt.show()