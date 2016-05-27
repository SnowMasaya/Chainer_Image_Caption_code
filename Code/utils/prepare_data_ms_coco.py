from pycocotools.coco import COCO
from PIL import Image
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "."))


class Prepare_Data_MS_COCO():

    def __init__(self, imgeDir, resizeImageDir, dataDir, dataType):
        self.ImageDir = imgeDir
        self.ResizeImageDir = resizeImageDir
        self.dataDir = dataDir
        self.dataType = dataType
        self.annFile = '%s/annotations 2/instances_%s.json' % (dataDir, dataType)

        # initialize COCO api for instance annotations
        self.coco = COCO(self.annFile)
        self.name_ids = {}
        self.img_dict = {}
        self.names = []
        self.ids = []

    def get_name_id(self):
        # display COCO categories and supercategories
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.names = [cat['name'] for cat in cats]
        self.ids = [cat['id'] for cat in cats]

        for i in range(len(self.names)):
            if self.ids[i] not in self.name_ids:
                self.name_ids.update({self.names[i]: self.ids[i]})
        self.__make_img_dict()

    def __make_img_dict(self):
        # get all images containing given categories, select one at random

        for name in self.names:
            catIds = self.coco.getCatIds(catNms=[name])
            imgIds = self.coco.getImgIds(catIds=catIds)
            for i in range(len(imgIds)):
                img = self.coco.loadImgs(imgIds[i])[0]
                if img["file_name"] not in self.img_dict:
                    self.img_dict.update({img["file_name"]: name})
        self.__resize_image_and_make_label_file()

    def __resize_image_and_make_label_file(self):
        for k, v in sorted(self.img_dict.items(), key=lambda x: x[0]):
            ImageFile = '%s/%s' % (self.ImageDir, k)
            pil_im = Image.open(ImageFile)
            out = pil_im.resize((255, 255))
            save_image = '%s/%s' % (self.ResizeImageDir, k)
            out.save(save_image)
            print(save_image + " " + str(self.name_ids[v]))
