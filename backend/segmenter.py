
import torch
from torchvision.models.segmentation import fcn_resnet101
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from dds_utils import Region, Results
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import glob
import os
import logging
import threading
from .segmenter_rpn import RPN
import cv2
import numpy as np

from .super_resoluter import SuperResoluter

import yaml
with open('dds_env.yaml', 'r') as f:
    dds_env = yaml.load(f.read())

class Segmenter(object):

    # Ensure Segmenter class is a singleton
    __instance = None
    __lock = threading.Lock()
    def __new__(cls):

        if Segmenter.__instance is None:
            with Segmenter.__lock:
                if Segmenter.__instance is None:
                    Segmenter.__instance = super(Segmenter, cls).__new__(cls)
        return Segmenter.__instance


    def __init__(self):

        self.logger = logging.getLogger("segmenter")
        handler = logging.NullHandler()
        self.logger.addHandler(handler)

        self.logger.info("loading fcn_resnet101...")
        self.model = fcn_resnet101(pretrained=True)

        # add forward hooks to get features
        self.features = []
        self.requires_features = False
        def append(tensor):
            if self.requires_features:
                self.features.append(tensor.detach())
        def hook(module):
            module.register_forward_hook(lambda m, i, o: append(o))
        hook(self.model.backbone.layer1)
        hook(self.model.backbone.layer2)
        hook(self.model.backbone.layer3)
        hook(self.model.backbone.layer4)
        self.model.register_forward_hook(lambda m,i,o: append(o['out']))

        self.model.eval().cuda()

        # image normalization
        self.im2tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # initialize RPN
        self.logger.info("loading rpn...")
        self.rpn = RPN(self)

        # import pdb; pdb.set_trace()

        # initialize super resoluter
        if dds_env['enable_cloudseg']:
            self.logger.info("loading super resoluter for cloudseg...")
            self.super_resoluter = SuperResoluter(16, 4)

    def transform(self, images):
        if not isinstance(images, torch.Tensor):
            images = torch.cat([self.im2tensor(i)[None,:,:,:].cuda() for i in images], dim=0)
        if dds_env['enable_cloudseg']:
            return self.super_resoluter(images)
        else:
            return images

    def infer(self, images, requires_grad = False, requires_features = False, requires_full = False):

        x = self.transform(images)
        self.features = []
        self.requires_features = requires_features

        if not requires_grad:
            func = torch.no_grad
        else:
            func = torch.enable_grad
        with func():
            if requires_full:
                return self.model(x)['out']
            else:
                return torch.max(self.model(x)['out'], 1)[1]


    def region_proposal(self, image, fid, resolution, k = dds_env['kernel_size'], topk = dds_env['num_sqrt'] * dds_env['num_sqrt']):

        def unravel_index(index, shape):
            out = []
            for dim in reversed(shape):
                out.append(index % dim)
                index = index // dim
            return tuple(reversed(out))

        def area_sum(grad):
            grad = torch.cumsum(torch.cumsum(grad, axis = 0), axis = 1)
            grad_pad = F.pad(grad, (k,k,k,k), value=-1)
            x, y = grad.shape
            grad_sum = grad[:, :] + grad_pad[0: x, 0:y] - grad_pad[k:x+k, 0:y] - grad_pad[0:x, k:y+k]
            return grad_sum

        def generate_regions(grad, results):
            x, y = grad.shape

            def set_zero(tensor, i, j):
                tensor[max(0,i-k+1) : min(i+1,x), max(0,j-k+1):min(j+1,y)] = 0

            for i in range(topk):

                index = unravel_index(torch.argmax(area_sum(grad)), grad.shape)
                index = [index[0].item(), index[1].item()]
                results.append(Region(fid, (index[1] - k + 1) / y, (index[0] - k + 1) / x, k / y, k / x, 1.0, 'pass', resolution))
                set_zero(grad, index[0], index[1])

        with torch.no_grad():
            pred = self.infer(image, requires_full = True)
            pred = F.softmax(pred, 1)
            # import pdb; pdb.set_trace()
            pred = -torch.sum(pred * torch.log(pred), dim = 1)
            grad = pred[0, :, :]

            '''
            # encourage edges
            pred = self.infer(image)
            pred = pred[0,:,:]
            pred[pred != 0] = 1
            pred = pred.cpu().data.numpy().astype(np.uint8)
            kernel = np.ones((16, 16), np.uint8)
            pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel)
            pred = pred - cv2.erode(pred, kernel)
            pred = torch.from_numpy(pred)
            grad[pred != 0] += 1
            '''

            grad = grad.type(torch.DoubleTensor)
            grad_ret = grad.clone()

            results = []
            generate_regions(grad, results)

        return results, grad_ret.cpu().numpy(), grad.cpu().numpy()


def run_rpn_inference(video, _,__,___, low_scale, low_qp, high_scale, high_qp, results_dir):

    segmenter = Segmenter()

    final_results = Results()

    dataset_root = Path(dds_env['dataset'])
    project_root = Path(dds_env['root'])
    lq_images_dir = dataset_root / f'{video}_{low_scale}_{low_qp}/src'
    assert lq_images_dir.exists()
    orig_images_dir = dataset_root / video / 'src'
    attention_dir = project_root / f'results_{video}' / f'{video}_dds_attention'
    attention_dir.mkdir(exist_ok=True)
    dilated_attention_dir = project_root / f'results_{video}' / f'{video}_dds_dilated_attention'
    dilated_attention_dir.mkdir(exist_ok=True)

    number_of_frames = len(glob.glob1(orig_images_dir, "*.png"))
    lq_images_path = [lq_images_dir / ('%010d.png' % i) for i in range(0, number_of_frames)]

    for idx, image_path in enumerate(lq_images_path):

        image = plt.imread(str(image_path))

        regions, attention, dilated_attention = segmenter.region_proposal([image], idx, low_scale)

        for region in regions:
            final_results.append(region)

        if dds_env['visualize'] and idx < 10:
            plt.clf()
            plt.figure(figsize=(16, 10))
            sns.heatmap(attention, cmap = 'Blues_r')
            plt.savefig(f"{attention_dir / ('%010d.png' % idx)}", bboxes_inches = 'tight')
            plt.close()
            plt.clf()
            plt.figure(figsize=(16, 10))
            sns.heatmap(dilated_attention, cmap = 'Blues_r')
            plt.savefig(f"{dilated_attention_dir / ('%010d.png' % idx)}", bboxes_inches = 'tight')
            plt.close()


        segmenter.logger.info(f'Region proposal for {image_path} completed.')

    bboxes_path = project_root / f'results_{video}'/ 'no_filter_combined_merged_bboxes'
    bboxes_path.mkdir(exist_ok=True)
    final_results.write(str(
        bboxes_path/
        f'{video}_mpeg_{low_scale}_{low_qp}'))

    bboxes_path = project_root / f'results_{video}'/ 'no_filter_combined_bboxes'
    bboxes_path.mkdir(exist_ok=True)
    final_results.write(str(
        bboxes_path/
        f'{video}_mpeg_{low_scale}_{low_qp}'))

    print('RPN Done')

