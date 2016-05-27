from pycocotools.coco import COCO
from PIL import Image
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "."))


class Prepare_Data_MS_COCO_Caption():

    def __init__(self, imgeDir, dataDir, dataType):
        self.ImageDir = imgeDir
        self.dataDir = dataDir
        self.dataType = dataType
        self.annFile = '%s/annotations 2/instances_%s.json' % (dataDir, dataType)
        self.captionFile = '%s/annotations 2/captions_%s.json' % (dataDir, dataType)

        # initialize COCO api for instance annotations
        self.coco_ann = COCO(self.annFile)
        self.coco_cap = COCO(self.captionFile)
        self.img_dict = {}
        self.captions_dict = {}
        self.names = []
        self.captions = []
        self.file_index2img = "index2img.txt"
        self.file_index2caption = "index2caption.txt"

    def get_name_id(self):
        # display COCO categories and supercategories
        cats = self.coco_ann.loadCats(self.coco_ann.getCatIds())
        self.names = [cat['name'] for cat in cats]

        self.__make_img_caption_dict()

    def __make_img_caption_dict(self):
        # get all images containing given categories, select one at random

        for name in self.names:
            catIds = self.coco_ann.getCatIds(catNms=[name])
            imgIds = self.coco_ann.getImgIds(catIds=catIds)
            for i in range(len(imgIds)):
                img = self.coco_ann.loadImgs(imgIds[i])[0]
                annIds = self.coco_cap.getAnnIds(imgIds=img["id"])[0]
                if img["id"] not in self.img_dict:
                    self.img_dict.update({img["id"]: img["file_name"]})
                if img["id"] not in self.captions_dict:
                    self.captions_dict.update({img["id"]: self.coco_cap.loadAnns(annIds)})
        self.__make_img_id_caption_file()

    def __make_img_id_caption_file(self):
        f_img = open(self.file_index2img, "w")
        f_caption = open(self.file_index2caption, "w")
        for k in sorted(self.img_dict.keys()):
            f_img.write(str(k) + "," + self.img_dict[k] + "\n")
            f_caption.write(str(k) + "\t" + self.captions_dict[k][0]["caption"] + "\n")
