import argparse
import os

import cv2

from model import PIIGANModel
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--image', default = 'examples/1', type = str,
                    help = 'The filename of image to be completed.')
parser.add_argument('--mask', default = 'examples/center_mask_128.png', type = str,
                    help = 'The filename of mask, value 255 indicates mask.')
parser.add_argument('--output', default = 'examples/output.png', type = str,
                    help = 'Where to write output.')
parser.add_argument('--checkpoint_dir',
                    default = 'model_logs/face_model', type = str,
                    help = 'The directory of tensorflow checkpoint.')

if __name__ == "__main__":
#     ng.get_gpus(0)
    args = parser.parse_args()

    model = PIIGANModel()

    image_list = os.listdir(args.image)
    image_list.sort()
    for i, single_image in enumerate(image_list):
      image_ = cv2.imread(os.path.join(args.image, single_image))
      image_ = np.expand_dims(image_, 0)
      if i == 0:
        image = image_
      else :
        image = np.concatenate((image, image_), axis = 0)
    image_to_write = image.reshape(-1, image.shape[1], 3)
#     image = cv2.imread(args.image)
    mask_ = cv2.imread(args.mask)
    mask_ = np.expand_dims(mask_, 0)
    for j in range(image.shape[0]):
      if j == 0:
        mask = mask_
      else :
        mask = np.concatenate((mask, mask_), axis = 0)

#     assert image.shape == mask.shape

    image_to_write = image.reshape(-1, image.shape[1], 3)
    mask_to_write = mask.reshape(-1, image.shape[1], 3)
    _, h, w, _ = image.shape
    grid = 8
    image = image[:, :h // grid * grid, :w // grid * grid, :]
    mask = mask[:, :h // grid * grid, :w // grid * grid, :]
    print('Shape of image: {}'.format(image.shape))

#     image = np.expand_dims(image, 0)
#     mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, mask], axis = 2)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config = sess_config) as sess:
        input_image = tf.constant(input_image, dtype = tf.float32)
        output = model.build_server_graph(input_image)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        z_code = model.build_server_graph__(input_image)
        #z = model.z

        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(args.checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)

        result0 = sess.run(output)
        result1 = sess.run(output)
        result2 = sess.run(output)
        result3 = sess.run(output)
        result4 = sess.run(output)
        result0 = result0.reshape(-1, image.shape[1], 3)
        result1 = result1.reshape(-1, image.shape[1], 3)
        result2 = result2.reshape(-1, image.shape[1], 3)
        result3 = result3.reshape(-1, image.shape[1], 3)
        result4 = result4.reshape(-1, image.shape[1], 3)




        (result4 == result3).all()
        result_list = np.concatenate([result0, result1, result2, result3, result4], axis = 1)
        image_mask = image_to_write + (255 * mask_to_write)
        result_ = np.concatenate([image_to_write, image_mask, result_list[:, :, ::-1]], axis = 1)
        cv2.imwrite(args.output, result_)
        
        
        print('Model loaded.')
        result_list = []
        for i in range(8):
          result = sess.run(output)
          result = result.reshape(-1, image.shape[1], 3)
          if i == 0:
            result_list = result
          else:
            result_list = np.concatenate([result_list, result], axis = 1)

#         cv2.imwrite(args.output, result_list[:, :, ::-1])
        mask_to_write = (mask_to_write > 127.5)

        image_mask = image_to_write + (255 * mask_to_write)
        result_ = np.concatenate([image_to_write, image_mask, result_list[:, :, ::-1]], axis = 1)
        cv2.imwrite(args.output, result_)
        print('Done')
