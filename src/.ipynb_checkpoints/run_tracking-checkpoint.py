from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import logging
import os
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import datasets.dataset.jde as datasets
from track import eval_seq
from tqdm import tqdm

logger.setLevel(logging.INFO)
import shutil 
def delete_empty_folders():
    vid_folder = '../demos'
    for f in tqdm(os.listdir(vid_folder)):
        if len(os.listdir(osp.join(vid_folder,f)))==0:
            shutil.move(osp.join(vid_folder,f), f)

def demo(opt, vid_path):
    # import pdb
    # pdb.set_trace()
    vid_name = vid_path.split('/')[-1].split('.')[0]

    result_root = opt.output_root if opt.output_root != '' else '.'
    result_root = osp.join(result_root, vid_name)
   
    if not  osp.exists(result_root):
        mkdir_if_missing( result_root)
      
    
        logger.info('Starting tracking...')
        print("Tracking video:{}".format(vid_path))

        dataloader = datasets.LoadVideo(vid_path, opt.img_size)
        result_filename = os.path.join(result_root, vid_name + '.txt')
        frame_rate = dataloader.frame_rate

        frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')
        eval_seq(opt, dataloader, 'mot', result_filename, save_dir=frame_dir, show_image=False, frame_rate=frame_rate)

    # if opt.output_format == 'video':
    #     output_video_path = osp.join(result_root,  vid_name + '.mp4')
    #     cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(osp.join(result_root, 'frame'), output_video_path)
    #     os.system(cmd_str)


if __name__ == '__main__':
    vid_folder = 'demo_vids'
    #delete_empty_folders()
    opt = opts().init()
    for f in tqdm(os.listdir(vid_folder)):
        for vid in os.listdir(osp.join(vid_folder,f)):
            vid_path = osp.join(vid_folder,f,vid)
            demo(opt, vid_path)