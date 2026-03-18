from distutils.command.config import config
import json
import os
import random

from torch.utils.data import Dataset
import torch
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption
import os
from torchvision.transforms.functional import hflip, resize

import math
import random
from random import random as rand

class DGM4_Dataset(Dataset):
    def __init__(self, config, ann_file, transform, max_words=30, is_train=True): 
        
        self.root_dir = '../../datasets'       
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self._apply_source_filter(config=config, is_train=is_train)
        if 'dataset_division' in config:
            self.ann = self.ann[:int(len(self.ann)/config['dataset_division'])]

        self.transform = transform
        self.max_words = max_words
        self.image_res = config['image_res']

        self.is_train = is_train

    @staticmethod
    def _normalize_source(source):
        if source is None:
            return None
        source = str(source).strip().lower().replace('-', '_').replace(' ', '_')
        alias = {
            'guardian': 'gardian',
            'gardin': 'gardian',
            'the_guardian': 'gardian',
            'usatoday': 'usa_today',
            'washingtonpost': 'washington_post',
            'washingtonpost.com': 'washington_post',
        }
        return alias.get(source, source)

    def _get_ann_source(self, ann):
        source_keys = ('source', 'news_source', 'publisher', 'media', 'domain', 'site')
        for key in source_keys:
            if key in ann and ann[key]:
                return self._normalize_source(ann[key])

        image_path = ann.get('image', '')
        path_parts = str(image_path).split('/')
        if 'origin' in path_parts:
            origin_idx = path_parts.index('origin')
            if origin_idx + 1 < len(path_parts):
                return self._normalize_source(path_parts[origin_idx + 1])
        return None

    def _apply_source_filter(self, config, is_train):
        if is_train:
            include_sources = config.get('train_sources', None)
            exclude_sources = config.get('train_exclude_sources', None)
        else:
            include_sources = config.get('val_sources', config.get('eval_sources', None))
            exclude_sources = config.get('val_exclude_sources', config.get('eval_exclude_sources', None))

        include_sources = [self._normalize_source(s) for s in include_sources] if include_sources else []
        exclude_sources = [self._normalize_source(s) for s in exclude_sources] if exclude_sources else []

        if not include_sources and not exclude_sources:
            return

        ann_before = len(self.ann)
        unknown_source = 0
        filtered_ann = []
        for ann in self.ann:
            source = self._get_ann_source(ann)
            if source is None:
                unknown_source += 1
                if include_sources:
                    continue
                filtered_ann.append(ann)
                continue

            if include_sources and source not in include_sources:
                continue
            if exclude_sources and source in exclude_sources:
                continue
            filtered_ann.append(ann)

        self.ann = filtered_ann
        split = 'train' if is_train else 'val/test'
        print(
            f"[DGM4_Dataset] source filter on {split}: {ann_before} -> {len(self.ann)} "
            f"(unknown_source={unknown_source}, include={include_sources}, exclude={exclude_sources})"
        )
        
    def __len__(self):
        return len(self.ann)

    def get_bbox(self, bbox):
        xmin, ymin, xmax, ymax = bbox
        w = xmax - xmin
        h = ymax - ymin
        return int(xmin), int(ymin), int(w), int(h)    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        img_dir = ann['image']    
        image_dir_all = f'{self.root_dir}/{img_dir}'

        try:
            image = Image.open(image_dir_all).convert('RGB')   
        except Warning:
            raise ValueError("### Warning: fakenews_dataset Image.open")   
                         
        W, H = image.size
        has_bbox = False
        try:
            x, y, w, h = self.get_bbox(ann['fake_image_box'])
            has_bbox = True
        except:
            fake_image_box = torch.tensor([0, 0, 0, 0], dtype=torch.float)

        do_hflip = False
        if self.is_train:
            if rand() < 0.5:
                # flipped applied
                image = hflip(image)
                do_hflip = True

            image = resize(image, [self.image_res, self.image_res], interpolation=Image.BICUBIC)
        image = self.transform(image)
            
        if has_bbox:
            # flipped applied
            if do_hflip:  
                x = (W - x) - w  # W is w0

            # resize applied
            x = self.image_res / W * x
            w = self.image_res / W * w
            y = self.image_res / H * y
            h = self.image_res / H * h

            center_x = x + 1 / 2 * w
            center_y = y + 1 / 2 * h

            fake_image_box = torch.tensor([center_x / self.image_res, 
                        center_y / self.image_res,
                        w / self.image_res, 
                        h / self.image_res],
                        dtype=torch.float)

        label = ann['fake_cls']
        caption = pre_caption(ann['text'], self.max_words)
        fake_text_pos = ann['fake_text_pos']

        fake_text_pos_list = torch.zeros(self.max_words)

        for i in fake_text_pos:
            if i<self.max_words:
                fake_text_pos_list[i]=1
        
                
        return image, label, caption, fake_image_box, fake_text_pos_list, W, H
