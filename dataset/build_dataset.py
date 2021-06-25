# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 11:02:34 2021

@author: A0H34750
"""

# Basic module
import time

import random
from pathlib import Path
import pickle
import math

import os

from zipfile import ZipFile
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Module progress bar
from tqdm import tqdm

# Module scikit-image
from skimage import io as skio
from skimage import draw as drw
# from skimage import measure

from sklearn.cluster import DBSCAN

# Hyperparameters

DATA_DIR_BASE = './ressources'
DATA_DIR_TARGET = './ressources/dataset/oring5'
DATA_DIR_SOURCE = './ressources/png'


def list_pns():
    '''
    Get part numbers from ZIP file

    Returns
    -------
    NUMPY ARRAY
        DESCRIPTION.

    '''

    zip_file = ZipFile(r'./ressources/BackupFuel_20210325.zip')
    d_f = pd.read_html(zip_file.read('Assembly.xls').decode("utf-8") + '</table>')[0]

    # Find position (i,j) of 'BE Code' in xls table (xls is just a text file with HTML table tag)
    s_1 = d_f.eq('BE Code').any(1)
    i = s_1.loc[s_1].index[0]
    s_2 = d_f.eq('BE Code').any(0)
    j = s_2.loc[s_2].index[0]

    # delete first 3 rows
    d_f = d_f.drop(range(0,i+1), axis=0)
    # remove all spaces
    d_f[j] = d_f[j].str.replace(' ', '')
    d_f[j] = d_f[j].str.replace('HR', '')

    return d_f[j].to_numpy()

def list_done(mydir,num):
    '''


    Parameters
    ----------
    mydir : STR
        directory.
    num : INT
        position in string.

    Returns
    -------
    LIST
        DESCRIPTION.

    '''
    mylist=[]
    deb = num
    end = deb + 5
    # get file
    for path in Path(mydir).rglob('*.png*'):
        mylist.append(path.stem[deb:end])

    return list(dict.fromkeys(mylist))

def list_done2(mydir,num):
    mylist=[]
    deb = num
    end = deb + 5
    # get file
    for path in tqdm(Path(mydir).rglob('*.png*'),total=593):
        im_ld = skio.imread(path)
        if len(im_ld.shape) == 3:
            mylist.append(path.stem[deb:end])

    return list(dict.fromkeys(mylist))


def num_to_rgb(val, max_val=3):
    '''
    Return RGB value from an integer

    Parameters
    ----------
    val : TYPE
        DESCRIPTION.
    max_val : TYPE, optional
        DESCRIPTION. The default is 3.

    Returns
    -------
    r_value : TYPE
        DESCRIPTION.
    g_value : TYPE
        DESCRIPTION.
    b_value : TYPE
        DESCRIPTION.

    '''
    i = (val * 255 / max_val)
    r_value = round(math.sin(0.024 * i + 0) * 127 + 128)
    g_value = round(math.sin(0.024 * i + 2) * 127 + 128)
    b_value = round(math.sin(0.024 * i + 4) * 127 + 128)
    return (r_value,g_value,b_value)


def segment_image(part_nb):
    # Hyperparameter
    DB_EPS = 25
    print('\nDBSCAN for {} at {}'.format(part_nb,time.ctime()))

    # get file
    for path in Path(DATA_DIR_SOURCE).rglob('*' + part_nb + '*.png*'):
        im = skio.imread(path)

    # get dataset (ie transform image matrix in 2 columns matrix with only black pixel coordinates
    Xsi = np.column_stack(np.where(im == 0))
    print('image dimension: {} -- number of pixels: {}'.format(im.shape,Xsi.shape))

    db = DBSCAN(eps=DB_EPS, min_samples=10).fit(Xsi)
    # core_samples_mask are points at boundary (not in center of cluster)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # build an image filled with white pixels
    si_img = np.full((im.shape[0],im.shape[1],3), (255,255,255))

    unique_labels = set(labels)
    # color all pixels from class with same color
    for k_l in unique_labels:
        class_member_mask = (labels == k_l)
        xy = Xsi[class_member_mask]
        si_img[xy[:,0],xy[:,1]] = num_to_rgb(k_l,len(unique_labels))

    # draw black rectangle around classes
    for k_l in unique_labels:
        if k_l != -1:
            class_member_mask = (labels == k_l)
            xy = Xsi[class_member_mask]
            rr, cc = drw.rectangle_perimeter(
                start=(np.amin(xy[:,0]), np.amin(xy[:,1])),
                end=(np.amax(xy[:,0]), np.amax(xy[:,1])),
                shape=si_img.shape
                )
            si_img[rr, cc] = (1,1,1)

    # # save image
    # skio.imsave(FILE_NAME.replace(".png", "segmented.png"),img)
    return Xsi, labels, im.shape


def plot_label(Xdata,labels,unique_labels):
    n_clusters_ = len(unique_labels) - (1 if -1 in unique_labels else 0)
    cmap = plt.cm.get_cmap("Spectral")
    colors = [cmap(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for kla, col in zip(unique_labels, colors):
        if kla== -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == kla)
        xy = Xdata[class_member_mask]

        plt.plot(xy[:, 1], -xy[:, 0], 'o', markerfacecolor=tuple(col),
                 markeredgecolor=tuple(col), markersize=1)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


# filter
def label_filter(Xdata,labels,dimg,density=0):
    processed_labels = []
    unique_labels = set(labels)
    for idx_label in unique_labels:
        if idx_label != -1:
            class_member_mask = (labels == idx_label)
            xy = Xdata[class_member_mask]
            hw = np.amax(xy, axis=0)-np.amin(xy, axis=0)
            cond_img_large = np.amin(hw) > 512
            cond_img_small = np.amax(0.5*np.asarray(dimg) - hw) > 0

            if cond_img_large and cond_img_small:
                cond_img_dense = int(100*xy.shape[0]/(hw[0]*hw[1])) > density
                if cond_img_dense:
                    processed_labels.append(idx_label)

    return processed_labels


def fillBackground(image,radius,mypattern,mydensity):

    # BLACK = (0,0,0)

    center = (int(image.shape[0]/2), int(image.shape[1]/2))

    if mypattern == 'noise':
        rr, cc = drw.disk(center, radius, shape=image.shape)
        # creation of index (same size of rr and cc)
        index = np.arange(len(rr))
        # mix index values
        random.shuffle(index)
        # resize index to put in black not all pixels
        mydensity = random.randint(1, 7)/10.0
        index = index[:int(len(index)*mydensity)]
        image[rr[index], cc[index]] = (0,0,0)

    elif mypattern == 'full':
        rr, cc = drw.disk(center, radius, shape=image.shape)
        image[rr, cc] = (0,0,0)

    elif mypattern == 'hash':
        space = random.randint(4, radius)
        theta = random.randint(1, 180)
        for step in range(0,2*radius,space):
            # calculate opposite side length (tg = opp / adj)
            opp = int(np.sqrt(radius**2 - (radius-step)**2))
            line1 = ((center[0]+opp,center[1]+step-radius),(center[0]-opp,center[1]+step-radius))
            # rotation for hash
            line1 = (rotate(line1[0],center,theta),rotate(line1[1],center,theta))
            rr, cc = drw.line(line1[0][0],line1[0][1],line1[1][0],line1[1][1])
            image[rr, cc] = (0,0,0)

    return image


def build_circle(radius,thicks,pattern,density):

    my_color = {
        'BLACK': (0,0,0),
        'GRAY': (127,127,127),
        'WHITE': (255,255,255)
        }

    # build image with 'transparent' background
    # 127 is transparent background
    im = np.full((2*radius+2*thicks-1,2*radius+2*thicks-1,3), my_color['GRAY'])

    center = (int(im.shape[0]/2), int(im.shape[1]/2))

    # draw circle with thickness
    rr, cc = drw.disk(center, radius+thicks, shape=im.shape)
    im[rr, cc] = my_color['BLACK']
    rr, cc = drw.disk(center, radius, shape=im.shape)
    im[rr, cc] = my_color['WHITE']

    return fillBackground(im,radius,pattern,density)


def find_rad_cent(frc__img):

    frc_const = {
        'NUM': 10,
        'RADIUS_MIN': 72,
        'RADIUS': 24}


    #radius = np.random.randint(RADIUS_MIN,int(min(h,w)/6),size=NUM)
    # Size identical and equal to 24
    # frc_const['RADIUS'] = random.randint(10,50)
    # radius = np.full(frc_const['NUM'],frc_const['RADIUS'])

    radius = np.random.randint(10,30,frc_const['NUM'])

    # remove some cols and rows from image img
    # keep only center area of original image
    frc__img[0:frc_const['RADIUS_MIN'],:] = (255,255,255)
    frc__img[frc__img.shape[0]-frc_const['RADIUS_MIN']:frc__img.shape[0]-1,:] = (255,255,255)
    frc__img[:,0:frc_const['RADIUS_MIN']] = (255,255,255)
    frc__img[:,frc__img.shape[1]-frc_const['RADIUS_MIN']:frc__img.shape[1]-1,:] = (255,255,255)

    rr,cc = np.where(np.all(frc__img == (0,0,0), axis=-1))
    idx = random.sample(range(0, len(rr)-1), frc_const['NUM'])
    center = [(rr[idx[k]],cc[idx[k]]) for k in range(frc_const['NUM'])]

    distances = np.zeros((frc_const['NUM'],frc_const['NUM']),int)
    rayons = np.zeros((frc_const['NUM'],frc_const['NUM']),int)

    for (i,j),_ in np.ndenumerate(distances):
        if i > j:
            rayons[i][j] = (radius[i] + radius[j])*1.2
            distances[i][j] = np.linalg.norm(np.asarray(center[i])-np.asarray(center[j]))

    suppr = []
    for (i,j),_ in np.ndenumerate(distances):
        if i > j:
            if distances[i][j] - rayons[i][j] < 0:
                suppr.append(i)
                suppr.append(j)

    suppr = list(dict.fromkeys(suppr))
    radius = [radius[i] for i, _ in enumerate(radius) if i not in suppr]
    center = [center[i] for i, _ in enumerate(center) if i not in suppr]



    return radius[:5],center[:5]


def pixelate_rgb(imgi, window):
    n, m = imgi.shape[0], imgi.shape[1]
    n, m = n - n % window, m - m % window
    imgo = np.full((n,m,3), (0,0,0))
    for x in range(0, n, window):
        for y in range(0, m, window):
            imgo[x:x+window,y:y+window] = imgi[x:x+window,y:y+window].mean(axis=(0,1))

    return imgo

def rotate(node,center,angle):
    angle = np.radians(angle)
    x = center[0] + (node[0]-center[0])*np.cos(angle) - (node[1]-center[1])*np.sin(angle)
    y = center[1] + (node[0]-center[0])*np.sin(angle) + (node[1]-center[1])*np.cos(angle)
    return (int(x),int(y))

def get_regions(im):

    # pixelisation et seuillage
    # --> remove small line / area
    # 235 = 255 - 20
    SIZE = (235,235,235)
    img1 = pixelate_rgb(im, 20)

    IDX = np.all(img1 <= SIZE, axis=-1)
    img1[IDX]=(0,0,0)
    img1[np.invert(IDX)]=(255,255,255)

    # 1st step
    # use background=120 to get all regions (even background)
    # 120 is not use so equivalent to no background
    # img_regions = measure.label(img1,background=120,connectivity=1)

    img_regions = img1

    # find background region
    # assumption: background region is the region at location (0,0)
    # this because we add margins

    back_level = img_regions[0,0,0]

    IDX = np.all(img_regions == [back_level,back_level,back_level], axis=-1)
    img_regions[IDX]=(255,255,255)
    img_regions[np.invert(IDX)]=(0,0,0)

    return img_regions


def add_margin(pil_img, size):
    width, height = pil_img.size
    new_width = size
    new_height = size
    left = int((size - width)/2)
    top = int((size - height)/2)
    result = Image.new(pil_img.mode, (new_width, new_height), (255,255,255))
    result.paste(pil_img, (left, top))
    return result


def completeOring(ima,rad,thick,motif,orientation):

    center = (int(ima.shape[0]/2), int(ima.shape[1]/2))
    s = (3,2)
    imte = np.full((s[0]*rad+2*thick-1,s[1]*rad+2*thick-1,3), (127,127,127))

    line1=((center[0],center[1]-(rad+thick-1)),((center[0]+2*rad+2),center[1]+(rad+thick-1)))
    rr,cc = drw.rectangle(line1[0], line1[1])
    if motif == 'full':
        imte[rr, cc] = (0,0,0)
    else:
        imte[rr, cc] = (255,255,255)
        for sens in [-1,1]:
            for step in range(0,thick):
                line1=(
                    (center[0],center[1]+sens*(rad+step)),
                    (center[0]+2*rad+2,center[1]+sens*(rad+step))
                    )
                rr, cc = drw.line(line1[0][0],line1[0][1],line1[1][0],line1[1][1])
                imte[rr, cc] = (0,0,0)

    for couleur in [(0,0,0),(255,255,255)]:
        rr,cc = np.where(np.all(ima == (couleur), axis=-1))
        imte[rr,cc] = couleur

    if orientation:
        imte=np.flipud(imte)
     # end algorithm

    return imte


def build_dataset(im_png,id_n,partn,imagr,isprint=None):

    file = 1
    landmarks = []

    radius,center = find_rad_cent(imagr)

    data_images = {
        '/PNGImages/': im_png,
        '/Masks/': np.full((im_png.shape[0],im_png.shape[1],3), (255,255,255))
            }

    base_name = "data_"+str(partn)+"_"+format(file, '04d')+"_"+format(id_n, '12d') + ".png"

    for rad,cen in zip(radius,center):
        landmarks.append({'center':cen, 'radius':rad})

        # orient = random.randint(0, 1)
        for motif,folder in zip(['noise','full'],['/PNGImages/','/Masks/']):
            thick=3
            ima = build_circle(rad,thick,motif,0.2)
            imb = ima
            # imb = completeOring(ima,rad,thick,motif,orient)

            for couleur in [(0,0,0),(255,255,255)]:

                rr,cc = np.where(np.all(imb == couleur, axis=-1))

                if rr.size != 0 and cc.size != 0:
                    cond1 = np.amax(rr+cen[0]-rad-1) < im_png.shape[0]
                    cond2 = np.amax(cc+cen[1]-rad-1) < im_png.shape[1]
                    if cond1 and cond2:
                        
                        data_images[folder][rr+cen[0]-rad-1,cc+cen[1]-rad-1] = couleur

                        # if padding
                        #img = add_margin(Image.fromarray(np.uint8(data_images[folder])),2000)
                        
                        # if not padding
                        img = Image.fromarray(np.uint8(data_images[folder]))

                        # saving
                        img.save(DATA_DIR_TARGET + folder + base_name,"PNG")


    if isprint is not None:
        _, ax = plt.subplots(2)
        ax[0].imshow(im_png, cmap='Greys')
        ax[1].imshow(imagr, cmap='Greys')
        plt.title('Label #: {}'.format(id_n))
        plt.show()

    return {'name':base_name, 'label':id_n, 'filenum':file, 'landmarks': landmarks}


def get_subimage_k(idx_label,Xbik,labels):
    class_member_mask = (labels == idx_label)
    xy = Xbik[class_member_mask]

    margin = 75
    hw = np.amax(xy, axis=0)-np.amin(xy, axis=0) + 1 + margin
    # build image with white pixels
    img_output = np.full((hw[0],hw[1],3), (255,255,255))
    # add black pixels to image
    rr = xy[:,0] - np.min(xy[:,0]) + int(margin/2)
    cc = xy[:,1] - np.min(xy[:,1]) + int(margin/2)

    img_output[rr,cc] = (0,0,0)

    return img_output


def save_result(myrecords):
    with open(DATA_DIR_TARGET + "/data.pkl","wb") as f_obj:
        pickle.dump(myrecords, f_obj)



def iter_sample_fast(iterable, samplesize):
    results = []
    iterator = iter(iterable)
    # Fill in the first samplesize elements:
    try:
        for _ in range(samplesize):
            results.append(next(iterator))
    except StopIteration as stopit:
        raise ValueError("Sample larger than population.") from stopit
    random.shuffle(results)  # Randomize their positions
    for i, v in enumerate(iterator, samplesize):
        r = random.randint(0, i)
        if r < samplesize:
            results[r] = v  # at a decreasing rate, replace random items
    return results


# ----------------------------------------------------------------------------

def main(stop):

    LOOP_STOP = 4000
    folder = '/png_labeled_clean_crop/'

    if not stop:

        # from image of regions
        # build dataset and place result in:
            # Masks
            # PNGImages

        records = []
        zut = Path(DATA_DIR_BASE + folder).rglob('*_crop.png*')
        # get filename list in random order

        filenames = iter_sample_fast(zut,2451) #2701 2271
        inc = 0
        for path in tqdm(filenames,total=LOOP_STOP-1):
            pn = path.stem[2:7]
            img = skio.imread(path)
            imgr = get_regions(img)
            # _, _ = plt.subplots()
            # plt.imshow(imgr, cmap='Greys')
            id_rand =  random.randrange(1, 999999999999)
            mydict = build_dataset(img,id_rand,pn,imgr,isprint=0)

            records.append(mydict)
            save_result(records)
            inc += 1

            # stop for loop after number LOOP_STOP
            if inc == LOOP_STOP:
                break

    else:

        # identify region in drawing
        # split region and save image of region to disk
        pns_all = list_pns()
        pns_done = list_done(DATA_DIR_BASE+folder,len('HR'))
        pns_done2 = list_done2(DATA_DIR_SOURCE,2)
        pns_done += pns_done2

        pns = [x for x in pns_all if x not in pns_done]

        pns.sort(reverse=True)

        for pn in tqdm(pns,total=len(pns)):
            # idenify regions (DBSCAN)
            X,labels,dim = segment_image(pn)
            unique_labels = set(labels)

            # keep only relevant labeled regions
            filtered_labels = label_filter(X,labels,dim,density=0)

            #plot_label(X,labels,unique_labels)
            plot_label(X,labels,filtered_labels)
            print('\nNumbers all labels / filtered: {}/{}'.format(
                len(unique_labels),
                len(filtered_labels)
                ))


            for k in filtered_labels:
                img = get_subimage_k(k,X,labels)
                base_name = 'HR'+str(pn)+"_"+format(k, '04d') + "_labeled.png"
                skio.imsave(DATA_DIR_BASE + folder + base_name,img.astype(np.uint8))


def center_crop(img, new_width, new_height):

    width, height = img.size

    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    center_cropped_img = img.crop((left, top, right, bottom))

    return center_cropped_img


def rename_high(mydir):
    mylist = []
    # get file
    inc = 0
    for path in Path(mydir).rglob('*.png*'):
        mylist.append(path.stem)
        img = skio.imread(path)
        cond1 = max(img.shape[0],img.shape[1]) >= 2000
        cond1 = True
        cond2 = '_labeled_high' not in path.stem
        if cond1 and cond2:
            print(inc,path.stem,img.shape)
            inc += 1

            new_file = os.path.join(mydir, path.stem+"_high.png")
            os.rename(path, new_file)

    return 0

def checkConsistency(mydir):
    myPNG = []
    myMASK= []

    if 0 == 1:
        for path in Path(DATA_DIR_BASE+'/png_labeled_clean_crop/').rglob('*.png*'):
            img = Image.open(path)
            img = center_crop(img,500,500)

            new_file = os.path.join(DATA_DIR_BASE+'/png_labeled_clean_crop/',path.stem+'_crop.png')
            img.save(new_file,"PNG")


    # get file
    for path in Path(mydir+'/PNGImages').rglob('*.png*'):
        myPNG.append(path.stem)

    for path in Path(mydir+'/Masks').rglob('*.png*'):
        myMASK.append(path.stem)

    print(len(myPNG))
    print(len(myMASK))

    for name in myPNG:

        if name not in myMASK:
            print(name)

    for name in myMASK:
        if name not in myPNG:
            print(name)

    return 0

#rename_high(DATA_DIR_BASE+'/png_labeled/')
checkConsistency(DATA_DIR_TARGET)
#main(False)
