from __future__ import print_function, division

import numpy as np
#import torchvision
#from torchvision import datasets, models, transforms
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import DataLoader, Dataset
from PIL import Image
#import matplotlib.pyplot as plt
import time
import os
import copy



OUTPUT_FPS = 5

def usage():
    print('wrong arguments!')

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

in_dir = None
out_dir = None
label_file = None

start_time = time.time()
prev_end_time = start_time


parser = argparse.ArgumentParser(description='Process video files into labeled frames.')
parser.add_argument('--in_dir', type=str, default='', help='directory of video files',required=True)
parser.add_argument('--out_dir', default='labeled_frames', help='output directory')
parser.add_argument('--log_dir', default='logs', help='Log dir [default: log]')
parser.add_argument('--label_file', type=str, help='*.tsv labels file generated from BORIS',required=True)
FLAGS = parser.parse_args()
print('argv:',sys.argv)

in_dir = FLAGS.in_dir
out_dir = FLAGS.out_dir
label_file = FLAGS.label_file
LOG_DIR = FLAGS.log_dir

    
class Label_dirs(NamedTuple):
    not_laying: str
    laying_not_nursing: str
    laying_nursing: str
        
class Video(NamedTuple):
    cap: cv2.VideoCapture
    ds_rate: int
    name: str
    elapsed_frames: int
        
def get_elapsed_time(stime):
    total_secs = int(time.time()-stime)
    hours = total_secs//(60*60)
    leftover_secs = total_secs % (60*60)
    minutes = leftover_secs//60
    seconds = leftover_secs % 60
    return hours,minutes,seconds
        
def print_elapsed_time(intro='',stime=start_time):
    global prev_end_time
    hours_sec,minutes_sec,seconds_sec = get_elapsed_time(prev_end_time)
    hours,minutes,seconds = get_elapsed_time(stime)
    log_string('------- ' + intro +  ' -------')
    log_string('------- run time: section: %d hours, %d minutes, %d seconds, total: %d hours, %d minutes, %d seconds -------\n' % (hours_sec, minutes_sec, seconds_sec, hours, minutes, seconds))
    prev_end_time = time.time()

        
def get_outfile_name(vid, frame_i):
    base_name = vid.name.split('.')
    a = datetime.datetime.strptime(base_name[0], '%Y%m%d-%H%M%S')
    a = a + datetime.timedelta(seconds=frame_i*1/OUTPUT_FPS)
    outfile_name = datetime.datetime.strftime(a,'%Y%m%d_%H%M%S_%f')[0:-5] #get rid of zero padding
    return outfile_name
    

def read_label_row(row,is_laying,is_nursing):
    catI = 5
    statusI = 8
    if row[catI] == 'LAYING':
        if row[statusI] == 'START':
            is_laying = True
        else:
            is_laying = False
    if row[catI] == 'NURSE':
        if row[statusI] == 'START':
            is_nursing = True
        else:
            is_nursing = False
    return is_laying,is_nursing
    
#display for jupyter notebook
def display_frame(frame):
    #display frame
    cv2.imshow('Frame',frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def open_new_video(vid_q,elapsed_frames):
    fname = vid_q.get()
    cap = cv2.VideoCapture(os.path.join(in_dir,fname))
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    ds_rate = round(fps/OUTPUT_FPS)
    return Video(cap,ds_rate,fname,elapsed_frames)

def read_frame(vid,vid_frame_i,vid_q ):
    if vid.cap.isOpened():
        ret, frame = vid.cap.read()
        #downsample
        for i in range(vid.ds_rate-1):
            if vid.cap.isOpened():
                vid.cap.grab()
    else:
        ret = False
    #if end of video, move to next video in queue
    if ret == False:
        frame = np.empty(shape=[0, 0])
        if not vid_q.empty():
            vid = open_new_video(vid_q,vid.elapsed_frames+vid_frame_i)
            vid_frame_i = 0
            return read_frame(vid,vid_frame_i,vid_q)
    vid_frame_i+=1
    return frame,vid,vid_frame_i

def write_frame(img,vid,frame_i,cur_label_dir):
    if img.size == 0:
        return
    scale = 0.1
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    out_name = get_outfile_name(vid, frame_i) + ".png"
    writepath = os.path.join(out_dir,cur_label_dir,out_name)
   # cv2.imwrite(writepath,resized,[int(cv2.IMWRITE_JPEG_QUALITY), 25])
    cv2.imwrite(writepath,resized,[int(cv2.IMWRITE_PNG_COMPRESSION),5])
    
def read_write_frame(vid,frame_i,vid_q,cur_label_dir):
    frame,vid,frame_i = read_frame(vid,frame_i,vid_q)
    write_frame(frame,vid,frame_i,cur_label_dir)
    video_time = datetime.timedelta(seconds=(vid.elapsed_frames+frame_i)*1/OUTPUT_FPS)
    return vid,frame_i,video_time

    
def main():
    #make label directories
    if not os.path.exists(out_dir): os.mkdir(out_dir)
    laying_dirs = [os.path.join(out_dir,'not_laying'),os.path.join(out_dir,'laying')]
    nursing_dirs = [os.path.join(laying_dirs[1],'not_nursing'),os.path.join(laying_dirs[1],'nursing')]
    label_dirs = Label_dirs(laying_dirs[0],nursing_dirs[0],nursing_dirs[1])
    for d in laying_dirs:
        if not os.path.exists(d): os.mkdir(d)
    for d in nursing_dirs:
        if not os.path.exists(d): os.mkdir(d)
            
            
    cur_label_dir = ''
    vid_q = queue.Queue()
    is_laying = False
    is_nursing = False
    labeling_started = False
    #empty video to start
    vid = Video(cv2.VideoCapture(),0,'',0)
    frame_i = 0
    video_time = datetime.timedelta(seconds=vid.elapsed_frames*1/OUTPUT_FPS)
    with open(label_file, newline='') as tsvfile:
        label_reader = csv.reader(tsvfile,delimiter='\t')
        for row in label_reader:
            if row[0] == "Player #1":
                vid_q.put(os.path.split(row[1])[1])
            if row[0] == "Time" and row[1] == "Media file path":
                labeling_started = True
                continue
            if labeling_started:
                label_time = datetime.timedelta(seconds=float(row[0]))
                 #sometimes boris logs start times a little late, so make sure they've started by checking for label dir
                while video_time < label_time and cur_label_dir:
                    vid,frame_i,video_time = read_write_frame(vid,frame_i,vid_q,cur_label_dir)
                is_laying,is_nursing = read_label_row(row,is_laying,is_nursing)
                log_string("row secs:" + row[0] + " row time:" + str(label_time) + "  video time: " + str(video_time) + " file name: " + vid.name)
                if is_laying==False:
                    cur_label_dir = label_dirs.not_laying
                elif is_nursing == False:
                    cur_label_dir = label_dirs.laying_not_nursing
                else:
                    cur_label_dir = label_dirs.laying_nursing
                    
        
        
if __name__ == "__main__":
    if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
    log_fname='log_label_pig_vids'+ datetime.datetime.now().strftime('%Y-%m-%d_%H-%M') +'.txt'
    LOG_FOUT = open(os.path.join(LOG_DIR, log_fname), 'w') 
    main()
    print_elapsed_time('end program')