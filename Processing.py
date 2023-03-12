import glob
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from util import load_itk, world_2_voxel, voxel_2_world 


RESIZE_SPACING = [1, 1, 1]
SAVE_FOLDER_image = '1_1_1mm_slices_lung'
SAVE_FOLDER_lung_mask = '1_1_1mm_slices_lung_masks'
SAVE_FOLDER_nodule_mask = '1_1_1mm_slices_nodule'


def seq(start, stop, step=1):
    n = int(round((stop - start)/float(step)))
    if n > 1:
        return([start + step*i for i in range(n+1)])
    else:
        return([])

def draw_circles(image,cands,origin,spacing):

	#make empty matrix, which will be filled with the mask
    image_mask = np.zeros(image.shape)

    #run over all the nodules in the lungs
    for ca in cands.values:

    	#get middel x-,y-, and z-worldcoordinate of the nodule
        radius = np.ceil(ca[4])/2
        coord_x = ca[1]
        coord_y = ca[2]
        coord_z = ca[3]
        image_coord = np.array((coord_x,coord_y,coord_z))

        #determine voxel coordinate given the worldcoordinate
        image_coord = world_2_voxel(image_coord,origin,spacing)


        #determine the range of the nodule
        noduleRangeX = seq(-radius, radius, spacing[0])
        noduleRangeY = seq(-radius, radius, spacing[1])
        noduleRangeZ = seq(-radius, radius, spacing[2])

        #create the mask
        for x in noduleRangeX:
            for y in noduleRangeY:
                for z in noduleRangeZ:
                    coords = world_2_voxel(np.array((coord_x+x,coord_y+y,coord_z+z)),origin,spacing)
                    if (np.linalg.norm(image_coord-coords) * RESIZE_SPACING[0]) < radius:
                        # print(np.round(coords[0]), np.round(coords[1]), np.round(coords[2]))
                        # print(image_mask.shape)
                        image_mask[int(np.round(coords[0])),int(np.round(coords[1])),int(np.round(coords[2]))] = int(1)
    
    return image_mask


def getUids(subset, cads):
    # imagesWithNodules = []
    uids = []
    subsetDir = '../luna16/LUNA16/subset{}'.format(subset) 	
    imagePaths = glob.glob("{}/*.mhd".format(subsetDir))
    print(len(imagePaths))
    for imagePath in imagePaths:
        imageName = os.path.split(imagePath)[1].replace('.mhd','')
        uids.append(imageName)
        # if len(cads[cads['seriesuid'] == imageName].index.tolist()) != 0: #dit moet efficienter kunnen!
            # imagesWithNodules.append(imagePath)    
    # return imagesWithNodules
    return uids

def createMask(i):
    luna_paths = glob.glob(f"../luna16/LUNA16/subset{i}/*.mhd")
    mask_path = f"../luna16/MASK/subset{i}/"
    cads = pd.read_csv("annotations.csv")

    if not os.path.exists(mask_path):
        os.makedirs(mask_path)

    for f in tqdm(luna_paths):
        img, origin, spacing = load_itk(f)

        imageName = os.path.split(f)[1].replace('.mhd','')
        image_cads = cads[cads['seriesuid'] == imageName]

        mask_a = draw_circles(img, image_cads, origin, spacing)
        mask_a = mask_a.T

        np.save(mask_path + imageName + ".npy", mask_a)


if __name__ == "__main__":
    # cads = pd.read_csv("annotations.csv")
    
    # for i in tqdm(range(10), desc="Subset"):
        # uids = getUids(i, cads)

    createMask(8)