#!/usr/bin/env python
from pycocotools.coco import COCO
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
from PIL import Image
from os import path
APP_ROOT = path.dirname( path.abspath( __file__ ) ) + "/../"


class Resize_Image():

    def __init__(self, imgeDir, resizeImageDir):
        self.ImageDir = imgeDir
        self.ResizeImageDir = resizeImageDir
        self.dataDir = APP_ROOT + "/Data/"
        self.dataType = 'val2014'
        self.annFile = '%s/annotations/instances_%s.json'\
                       % (self.dataDir, self.dataType)

        # initialize COCO api for instance annotations
        self.coco = COCO(self.annFile)

        # display COCO categories and supercategories
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self.names = [cat['name'] for cat in self.cats]
        self.ids = [cat['id'] for cat in self.cats]
        self.name_ids = {}
        # get all images containing given categories, select one at random
        self.img_dict = {}

    def resize_image(self):

        for i in range(len(self.names)):
            if self.ids[i] not in self.name_ids:
                self.name_ids.update({self.names[i]: self.ids[i]})
        self.__image_dict_update()

    def __image_dict_update(self):

        for name in self.names:
            catIds = self.coco.getCatIds(catNms=[name])
            imgIds = self.coco.getImgIds(catIds=catIds)
            for i in range(len(imgIds)):
                img = self.coco.loadImgs(imgIds[i])[0]
                if img["file_name"] not in self.img_dict:
                    self.img_dict.update({img["file_name"]: name})
        self.__output_resize_images()

    def __output_resize_images(self):

        for k, v in sorted(self.img_dict.items(), key=lambda x: x[0]):
            ImageFile = '%s/%s' % (self.ImageDir, k)
            pil_im = Image.open(ImageFile)
            out = pil_im.resize((255, 255))
            save_image = '%s/%s' % (self.ResizeImageDir, k)
            out.save(save_image)
            print(save_image + " " + str(self.name_ids[v]))
