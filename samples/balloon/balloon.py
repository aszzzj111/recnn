"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------
训练配置参数
train
--dataset=E:/fffffile/Mask_RCNN-2.1/balloon/
--weights=E:/fffffile/Mask_RCNN-2.1/mask_rcnn_coco.h5
测试配置参数
splash
--weights=E:/fffffile/Mask_RCNN-2.1/samples/balloon/logs/balloon20230503T1216/mask_rcnn_balloon_0030.h5
--image=E:\fffffile\Mask_RCNN-2.1\balloon\val/16335852991_f55de7958d_k.jpg

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# Root directory of the project
# 项目的根目录
ROOT_DIR = os.getcwd()
if ROOT_DIR.endswith("samples/balloon"):
    # Go up two levels to the repo root 向上两级到达回购根
    ROOT_DIR = os.path.dirname(os.path.dirname(ROOT_DIR))

# Import Mask RCNN
sys.path.append(ROOT_DIR)
from config import Config
import utils
import model as modellib

# Path to trained weights file 训练权重文件的路径
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
# 保存日志和模型检查点的目录（如果未提供，通过命令行参数--logs提供）
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class BalloonConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    玩具数据集上的训练配置。从基本Config类派生，并覆盖某些值。
    """
    # Give the configuration a recognizable name 为配置提供一个可识别的名称
    NAME = "balloon"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    # 我们使用12GB内存的GPU，可以容纳两张图像。如果使用较小的GPU，请向下调整。
    IMAGES_PER_GPU = 1

    # Number of classes (including background) 类别数量（包括背景）
    NUM_CLASSES = 1 + 1  # Background + baloon

    # Number of training steps per epoch  每个历元的训练步骤数
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence 以<90%的置信度跳过检测
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset 数据集
############################################################

class BalloonDataset(utils.Dataset):

    def load_balloon(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset. 加载Balloon数据集的子集。
        dataset_dir: Root directory of the dataset.  数据集的根目录。
        subset: Subset to load: train or val  要加载的子集：train或val
        """
        # Add classes. We have only one class to add. 添加类。我们只有一个类要添加。
        self.add_class("balloon", 1, "balloon")

        # Train or validation dataset? 训练还是验证数据集？
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations  加载注释
        # VGG Image Annotator saves each image in the form（VGG图像注释器将每个图像保存在表单中）:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region  我们最关心的是每个区域的x和y坐标
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images. VIA工具将图像保存在JSON中，即使它们没有任何注释。跳过未标记的图像。
        annotations = [a for a in annotations if a['regions']]

        # Add images 添加图片
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            # 获取构成每个对象实例轮廓的多边形的点的x、y坐标。shape_attributes中有存储（请参阅上面的json格式）
            polygons = [r['shape_attributes'] for r in a['regions'].values()]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            # load_mask（）需要图像大小才能将多边形转换为遮罩。不幸的是，VIA没有将其包含在JSON中，所以我们必须读取图像。这只是可管理的，因为数据集很小。
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "balloon",
                image_id=a['filename'],  # use file name as a unique image id 使用文件名作为唯一的图像id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.为图像生成实例掩码。
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance. 形状为[高度、宽度、实例计数]的布尔数组，每个实例有一个掩码。
        class_ids: a 1D array of class IDs of the instance masks. 实例掩码的类ID的1D阵列。
        """
        # If not a balloon dataset image, delegate to parent class.
        # 如果不是气球数据集图像，则委托给父类。
        image_info = self.image_info[image_id]
        if image_info["source"] != "balloon":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape 将多边形转换为形状的位图遮罩
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            # 获取多边形内像素的索引并将其设置为1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        # 返回掩码和每个实例的类ID数组。由于我们只有一个类ID，所以我们返回一个1数组
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image.返回图像的路径。"""
        info = self.image_info[image_id]
        if info["source"] == "balloon":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset. 训练数据集。
    dataset_train = BalloonDataset()
    dataset_train.load_balloon(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset 验证数据集
    dataset_val = BalloonDataset()
    dataset_val.load_balloon(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***这个培训计划就是一个例子。根据您的需求更新
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    # 由于我们使用的是一个非常小的数据集，并且从COCO训练的权重开始，我们不需要训练太久。此外，不需要训练所有层，只需要头部就可以了。
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.应用颜色飞溅效果。
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.  返回结果图像。
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though. 制作图像的灰度副本。不过，灰度副本仍然有3个RGB通道。
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # We're treating all instances as one, so collapse the mask into one layer 制作图像的灰度副本。不过，灰度副本仍然有3个RGB通道。
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    # Copy color pixels from the original color image where mask is set 从设置遮罩的原始彩色图像中复制彩色像素
    if mask.shape[0] > 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video? 图像还是视频？
    if image_path:
        # Run model detection and generate the color splash effect 运行模型检测并生成颜色飞溅效果
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture 视频采集
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer  定义编解码器并创建视频编写器
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image 读取下一张图像
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB OpenCV以BGR形式返回图像，转换为RGB
                image = image[..., ::-1]
                # Detect objects 检测对象
                r = model.detect([image], verbose=0)[0]
                # Color splash  颜色飞溅
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video RGB->BGR将图像保存到视频
                splash = splash[..., ::-1]
                # Add image to video writer 将图像添加到视频编写器
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training 训练
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments 分析命令行参数
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments  验证参数
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations 配置
    if args.command == "train":
        config = BalloonConfig()
    else:
        class InferenceConfig(BalloonConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            # 将批量大小设置为1，因为我们将一次对一个图像进行inference。批次大小=GPU_COUNT*IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model 创建模型
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load  选择要加载的权重文件
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file  下载权重文件
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights  查找上次训练的重量
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights  从ImageNet训练的权重开始
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights  加载权重
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes 排除最后一层，因为它们需要匹配数量的类
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        # model.load_weights(weights_path, by_name=True)
         model.load_weights(weights_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask"])


    # Train or evaluate 训练或者评估
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
