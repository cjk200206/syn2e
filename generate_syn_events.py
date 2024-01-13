import argparse
import os
import esim_torch
import numpy as np
import glob
import cv2
import tqdm
import torch
from event_corner.event_corner import frame_corner_tube, judge_event_corner, save_corners

from syn.syn_test import make_syn_frames
from upsampling.utils import Upsampler



def is_valid_dir(subdirs, files):
    return len(subdirs) == 1 and len(files) == 1 and "timestamps.txt" in files and "imgs" in subdirs


def process_dir(outdir, indir, args):
    print(f"Processing folder {indir}... Generating events in {outdir}")
    os.makedirs(outdir, exist_ok=True)

    # constructor
    esim = esim_torch.ESIM(args.contrast_threshold_negative,
                           args.contrast_threshold_positive,
                           args.refractory_period_ns)

    timestamps = np.genfromtxt(os.path.join(indir, "timestamps.txt"), dtype="float64")
    timestamps_ns = (timestamps * 1e9).astype("int64")
    timestamps_ns = torch.from_numpy(timestamps_ns).cuda()

    image_files = sorted(glob.glob(os.path.join(indir, "imgs", "*.png")))
    
    pbar = tqdm.tqdm(total=len(image_files)-1)
    num_events = 0

    counter = 0
    for image_file, timestamp_ns in zip(image_files, timestamps_ns):
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        log_image = np.log(image.astype("float32") / 255 + 1e-5)
        log_image = torch.from_numpy(log_image).cuda()

        sub_events = esim.forward(log_image, timestamp_ns)

        # for the first image, no events are generated, so this needs to be skipped
        if sub_events is None:
            continue
        num_events += len(sub_events.T)

        # do something with the events
        event_dict = list(sub_events.T.cpu().numpy())
        np.savetxt(os.path.join(outdir, "%010d.txt" % counter), event_dict,fmt='%s')
        pbar.set_description(f"Num events generated: {num_events}")
        pbar.update(1)
        counter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser("""Generate events from a high frequency video stream""")
    parser.add_argument("--contrast_threshold_negative", "-cn", type=float, default=0.2)
    parser.add_argument("--contrast_threshold_positive", "-cp", type=float, default=0.2)
    parser.add_argument("--refractory_period_ns", "-rp", type=int, default=0)
    args = parser.parse_args()

    ##step1 生成帧图像
    img_root = "datasets/syn_test/img"
    points_root = "datasets/syn_test/points"   
    upsample_root = "datasets/syn_test/upsampled"
    events_root = "datasets/syn_test/events"
    event_corner_root = "datasets/syn_test/event_corners"

    # make_syn_frames(img_root,points_root,100)

    # ##step1.1 加入fps文件
    # fps_file = os.path.join(img_root,"fps.txt")
    # with open(fps_file,"w+") as f: 
    #     f.write('25') #帧率

    # ##step2 生成上采样
    # upsampler = Upsampler(input_dir=img_root, output_dir=upsample_root)
    # upsampler.upsample_new()

    # ##step3 生成事件
    # print(f"Generating events with cn={args.contrast_threshold_negative}, cp={args.contrast_threshold_positive} and rp={args.refractory_period_ns}")

    # for path, subdirs, files in os.walk(upsample_root):
    #     if is_valid_dir(subdirs, files):
    #         output_folder = os.path.join(events_root, os.path.relpath(path,upsample_root))

    #         process_dir(output_folder, path, args)

    ##step4 检测事件角点
    for dirpath,subdirs,filenames in os.walk(points_root):
        if len(subdirs) == 0 and len(filenames) != 0: #判断文件夹，当只含txt文件时，即最里侧文件夹
            frame_corner_dir = dirpath
            events_dir = os.path.join(events_root,os.path.relpath(dirpath,points_root))
            event_corner_dir = os.path.join(event_corner_root,os.path.relpath(dirpath,points_root))

            print(str(event_corner_dir)+' processing...') 
            tubes = frame_corner_tube(frame_corner_dir)
            event_corners = judge_event_corner(tubes,events_dir)
            save_corners(event_corners,event_corner_dir)
            print(str(event_corner_dir)+' finished!') 
        




