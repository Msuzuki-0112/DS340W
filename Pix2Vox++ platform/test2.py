import logging
import matplotlib
import multiprocessing as mp
import numpy as np

# Fix problem: no $DISPLAY environment variable
matplotlib.use('Agg')
from PIL import Image


from config import cfg
import torch.utils.data
import os
import utils.binvox_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils
import cv2
from datetime import datetime as dt
from tensorboardX import SummaryWriter
from time import time
from collections import OrderedDict
from core.test import test_net
from models.encoder import Encoder
from models.decoder import Decoder
from models.refiner import Refiner
from models.merger import Merger

epoch_idx = 1
# Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
torch.backends.cudnn.benchmark = True
encoder = Encoder(cfg)
decoder = Decoder(cfg)
refiner = Refiner(cfg)
merger = Merger(cfg)
if torch.cuda.is_available():
    encoder = torch.nn.DataParallel(encoder).cuda()
    decoder = torch.nn.DataParallel(decoder).cuda()
    refiner = torch.nn.DataParallel(refiner).cuda()
    merger = torch.nn.DataParallel(merger).cuda()
cfg.CONST.WEIGHTS = 'Pix2Vox-A-ShapeNet.pth'
print('[INFO] %s Loading weights from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
checkpoint = torch.load(cfg.CONST.WEIGHTS)
epoch_idx = checkpoint['epoch_idx']
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])
if cfg.NETWORK.USE_REFINER:
    refiner.load_state_dict(checkpoint['refiner_state_dict'])
if cfg.NETWORK.USE_MERGER:
    merger.load_state_dict(checkpoint['merger_state_dict'])

encoder.eval()
decoder.eval()
refiner.eval()
merger.eval()

img1_path = './images/1_replace.png'
img1_np = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
sample = np.array([img1_np])

IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
test_transforms = utils.data_transforms.Compose([
    utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
    utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
    utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
    utils.data_transforms.ToTensor(),
])

rendering_images = test_transforms(rendering_images=sample)
rendering_images = rendering_images.unsqueeze(0)

with torch.no_grad():
    # Get data from data loader
    rendering_images = utils.network_utils.var_or_cuda(rendering_images)

    # Test the encoder, decoder, refiner and merger
    image_features = encoder(rendering_images)
    raw_features, generated_volume = decoder(image_features)

    if cfg.NETWORK.USE_MERGER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_MERGER:
        print("Using Merger and Refiner")
        generated_volume = merger(raw_features, generated_volume)
    else:
        generated_volume = torch.mean(generated_volume, dim=1)

    if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
        generated_volume = refiner(generated_volume)

    generated_volume = generated_volume.squeeze(0)

    img_dir = './sample_images'
    gv = generated_volume.cpu().numpy()
    rendering_views = utils.binvox_visualization.get_volume_views(gv, os.path.join(img_dir),
                                                                  epoch_idx)