from PIL import Image
import numpy as np
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt
from normxcorr2 import normxcorr2

#define find offset function for beads file
def find_offset(path):
    img = Image.open(path)
    images = []
    for i in range(img.n_frames):
        img.seek(i)
        images.append(np.array(img))
        beads = np.array(images)
    beadsavg=np.mean(beads, axis=0)
    #initialize offsets; cut the edge for donor(20) and acceptor(10) channels
    offsetx = 0
    offsety = 0
    donor = beadsavg[(20):int(beadsavg.shape[0]-20),(20):int(beadsavg.shape[1]/2-20)]
    acceptor = beadsavg[10:int(beadsavg.shape[0]-10),int(beadsavg.shape[1]/2+10):int(beadsavg.shape[1]-10)]
    #find cross correlation array, size should be D+A-1
    cc = normxcorr2(donor,acceptor)
    #output the offset values, since we expect donor locate in tbe middle of acceptor, so -10; +1 since python starts at 0
    offsetx = np.where(cc==np.max(cc))[1][0]-donor.shape[1]-10+1
    offsety = np.where(cc==np.max(cc))[0][0]-donor.shape[0]-10+1
    return print('x offset = '+str(offsetx)+' and y offset = '+str(offsety))


#define read .tiff data file
def read_tiff(path):
    img = Image.open(path)
    images = []
    for i in range(img.n_frames):
        img.seek(i)
        images.append(np.array(img))
        data = np.array(images)
    return data
