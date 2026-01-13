import os
import shutil
from os.path import join
import xml.etree.ElementTree as ET

SOURCE_DIR = '/media/vision/storage1/Datasets/UA_DETRAC/UA_DETRAC'
DESTINATION_DIR = '/media/vision/storage1/Datasets/UA_DETRAC/sorted'

H_RESCALE = 540/1080
W_RESCALE = 960/1920

# sequences = []
# images_path = join(SOURCE_DIR, 'images')
# splits = os.listdir(images_path) # train and val dirs
# for split in splits:
#     split_images_path = join(images_path, split)
#     split_imgnames = os.listdir(split_images_path)

#     if not os.path.exists(join(DESTINATION_DIR, split)):
#         os.makedirs(join(DESTINATION_DIR, split))
#     for imgname in split_imgnames:
#         seqname = "_".join(imgname.split('_')[:2])
#         if seqname not in sequences:
#             sequences.append(seqname)
#             if not os.path.exists(join(DESTINATION_DIR, split, seqname, 'img1')):
#                 os.makedirs(join(DESTINATION_DIR, split, seqname, 'img1'))
        
#         imgpath = join(split_images_path, imgname)
#         shutil.copy(imgpath, join(DESTINATION_DIR, split, seqname, 'img1', imgname))

sequences = []
xmls_path = join(SOURCE_DIR, 'good_labels')
splits = os.listdir(xmls_path) # train and val dirs
classes_dict = {
    'car': 1,
    'van': 2,
    'bus': 3,
    'others': 4
}
for split in splits:
    split_xmls_path = join(xmls_path, split)
    split_xmlnames = os.listdir(split_xmls_path)

    if not os.path.exists(join(DESTINATION_DIR, split)):
        os.makedirs(join(DESTINATION_DIR, split))

    for xmlname in split_xmlnames:
        sequence_gt = []
        seqname = xmlname.split('.')[0]
        target_gt_filepath = join(DESTINATION_DIR, split, seqname, 'gt', 'gt.txt')
        print('Processing', seqname)

        if seqname not in sequences:
            sequences.append(seqname)
            if not os.path.exists(join(DESTINATION_DIR, split, seqname, 'gt')):
                os.makedirs(join(DESTINATION_DIR, split, seqname, 'gt'))
        xmlpath = join(split_xmls_path, xmlname)

        tree = ET.parse(xmlpath)
        root = tree.getroot()
        seq_name = root.get('name')  # video sequence name (subdirectory name)
        ignor_region = root.find('ignored_region')

        ignore_boxes = []
        for box_info in ignor_region.findall('box'):
            box = [int(0),                        # obj id
                   float(box_info.get('left')) * W_RESCALE,
                   float(box_info.get('top')) * H_RESCALE,
                   float(box_info.get('width')) * W_RESCALE,
                   float(box_info.get('height')) * H_RESCALE,
                   float(0),                        # conf
                   float(-1),
                   float(-1)]
            ignore_boxes.append(box)


        for frame in root.findall('frame'):
            frame_num = int(frame.get('num'))
            target_list = frame.find('target_list')
            targets = target_list.findall('target')

            for ignored_box in ignore_boxes:
                ignored_box_line = [frame_num] + ignored_box
                ignored_box_line = [str(el) for el in ignored_box_line]
                sequence_gt.append(",".join(ignored_box_line) + "\n")

            for target in targets:
                target_id = int(target.get('id'))

                bbox_info = target.find('box')
                bbox_left = float(bbox_info.get('left')) * W_RESCALE
                bbox_top = float(bbox_info.get('top')) * H_RESCALE
                bbox_width = float(bbox_info.get('width')) * W_RESCALE
                bbox_height = float(bbox_info.get('height')) * H_RESCALE

                attr_info = target.find('attribute')
                vehicle_type = str(attr_info.get('vehicle_type'))
                trunc_ratio = float(attr_info.get('truncation_ratio'))

                target_line = [
                    frame_num,
                    target_id,
                    bbox_left,
                    bbox_top,
                    bbox_width,
                    bbox_height,
                    1,
                    classes_dict[vehicle_type],
                    trunc_ratio
                ]
                target_line = [str(el) for el in target_line]
                sequence_gt.append(','.join(target_line) + '\n')

        with open(target_gt_filepath, "w") as file:
            file.writelines(sequence_gt)
        # print('xml name:', xmlname, 'target gt file:', target_gt_filepath)

        # shutil.copy(imgpath, join(DESTINATION_DIR, split, seqname, 'img1', imgname))