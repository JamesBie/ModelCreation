import os
import glob
import pandas as pd
import tensorflow as tf
import xml.etree.ElementTree as ET

from object_detection.utils import dataset_util

name_to_id = {'hold':1, 'volume':2}

def create_tf(file):
  # TODO(user): Populate the following variables from your example.
  height = file[0][2] # Image height
  width = file[0][1] # Image width
  filename = file[0][0] # Filename of the image. Empty if image is not from file
  encoded_image_data = None # Encoded image bytes
  image_format = b'jpeg' # b'jpeg' or b'png'
  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box
                 # (1 per box)
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box
                 # (1 per box)
  classes_text = [] # List of string class name of bounding box (1 per box)
  classes = [] # List of integer class id of bounding box (1 per box)

  path = os.path.join(os.getcwd(), "../Rock_Hold_Dataset/Done/"+filename)
  with tf.io.gfile.GFile(path, 'rb') as fid:
    encoded_jpg = fid.read()

  for item in file:
    print(item)
    xmins.append(float(item[4])/width)
    xmaxs.append(float(item[6])/width)
    ymins.append(float(item[5])/height)
    ymaxs.append(float(item[7])/height)
    classes_text.append(item[3].encode('utf8'))
    classes.append(name_to_id[item[3]])
      

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example


def xml_to_tfrecord(path):
    print("path is: ", path)
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        print(xml_file)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        print(root.tag)


        path = os.path.join(os.getcwd(), "records/"+root.find('filename').text+".record")
        # print(path)
        writer = tf.io.TFRecordWriter(path)
        for object_tag in root.findall('object'):
            print(object_tag)
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     object_tag [0].text,
                     int(object_tag [4][0].text),
                     int(object_tag [4][1].text),
                     int(object_tag [4][2].text),
                     int(object_tag [4][3].text)
                     )
            print(value)
            xml_list.append(value)
        print("done current xml")
        print(xml_list)

        tf_example = create_tf(xml_list)
        print('tf example:  ')
        print(tf_example)
        writer.write(tf_example.SerializeToString())

        writer.close()
        xml_list=[]


    # column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    # xml_df = pd.DataFrame(xml_list, columns=column_name)
    # print(xml_df)
    # return xml_df



def main():
    PATH = "../Rock_Hold_Dataset/Done"

    xml_to_tfrecord(PATH)
    # xml_df.to_csv('hold_labels.csv', index=None)
    print('Successfully converted xml')


main()