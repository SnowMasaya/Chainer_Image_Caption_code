# coding: utf-8


class ExcludeGrayImage():

    def __init__(self, original_image_list, gray_image_list,
                 exclude_image_list):
        self.original_image_list = open(original_image_list, "r")
        self.gray_image_list = open(gray_image_list)
        self.exclude_image_list = open(exclude_image_list, "w")
        self.__gray_image_name = []
        self.__gray_image_name_list = []

    def exclude_gray_image(self, separator="\t"):
        self.__get_gray_Image()
        self.__write_exclude_gray_image(separator)

    def __get_gray_Image(self):
        # Get Gray Image
        gray_img_name = self.gray_image_list.read()
        self.gray_img_name_list = gray_img_name.split("\n")

    def __write_exclude_gray_image(self, separator):
        for line in self.original_image_list:
            line = line.strip()
            catID, image_file_name = line.split(separator)
            if image_file_name not in self.gray_img_name_list:
                self.exclude_image_list.write(line+"\n")
