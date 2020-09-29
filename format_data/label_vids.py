import cv2
import numpy as np
import os
import fnmatch
import csv
from typing import NamedTuple
import datetime
import time
import getopt
import sys




OUTPUT_FPS = 5

def usage():
    print('wrong arguments!')

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

indir = None
outdir = None
label_file = None
vid_list = None

LOG_DIR='logs'
#LOG_FOUT = None
start_time = time.time()
prev_end_time = start_time


def set_opts(args=sys.argv[1:]):
    global indir, outdir, vid_list, label_file, LOG_DIR, LOG_FOUT
    try:
        opts, args = getopt.getopt(args, "i:o:l:p:g:", ['in_dir=','out_dir=', 'label_file=', 'vid_path_list=','log_dir='])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)
    for o, a in opts:
        if o in ("-i", "--in_dir"):
            indir=a
        elif o in ("-o", "--out_dir"):
            outdir=a
        elif o in ("-l", "--label_file"):
            label_file = a
        elif o in ("-g", "--log_dir"):
            LOG_DIR=a
        elif o in ("-p", "--vid_path_list"):
            vid_list = get_vid_list(a)
            
        else:
            assert False, "unhandled option"
    if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
    log_fname='log_label_pig_vids'+ datetime.datetime.now().strftime('%Y-%m-%d_%H-%M') +'.txt'
    LOG_FOUT = open(os.path.join(LOG_DIR, log_fname), 'w')     

    
class Label_dirs(NamedTuple):
    not_laying: str
    laying_not_nursing: str
    laying_nursing: str
        
class vid_info(NamedTuple):
    name: str
    start: float
    end: float
    index: int
        
class Video(NamedTuple):
    cap: cv2.VideoCapture
    ds_rate: int
    info: vid_info
        
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

        
def get_vid_list(vid_path_list):
    i = 0
    time_so_far = 0
    vid_list = []
    with open(vid_path_list,newline='') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in file_reader:
            vidname = row[0]
            lenstr = row[1].split(':')
            vid_secs = int(lenstr[1])*60+float(lenstr[2])
            vid_list.append(vid_info(vidname,time_so_far,time_so_far+vid_secs,i))
            i+=1
            time_so_far+=vid_secs
    return vid_list
        
def get_outfile_name(vid, frame_i):
    base_name = vid.info.name.split('.')
    a = datetime.datetime.strptime(base_name[0], '%Y%m%d-%H%M%S')
    a = a + datetime.timedelta(seconds=frame_i*1/OUTPUT_FPS)
    outfile_name = datetime.datetime.strftime(a,'%Y%m%d_%H%M%S_%f')[0:-5] #get rid of zero padding
    return outfile_name
    

def read_label_row(row,is_laying,is_nursing):
    timeI = 0
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
    
def display_frame(frame):
    #display frame
    cv2.imshow('Frame',frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def open_new_video(vid_i):
    info = vid_list[vid_i]
    cap = cv2.VideoCapture(os.path.join(indir,info.name))
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    ds_rate = round(fps/OUTPUT_FPS)
    frame_time = 0
    return Video(cap,ds_rate,info)

def read_frame(vid,frame_i ):
    if vid.cap.isOpened():
        ret, frame = vid.cap.read()
        frame_i +=1
        for i in range(vid.ds_rate-1):
            if vid.cap.isOpened():
                vid.cap.grab()
    else:
        ret = False
    if ret == False:
        if vid.info.index+1 < len(vid_list):
            vid = open_new_video(vid.info.index+1)
            frame_i = 0
            return read_frame(vid,frame_i)
    return ret,frame,vid, frame_i

def write_file(img,vid,frame_i,cur_label_dir):
    scale = 0.1
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    out_name = get_outfile_name(vid, frame_i) + ".png"
    writepath = os.path.join(outdir,cur_label_dir,out_name)
   # cv2.imwrite(writepath,resized,[int(cv2.IMWRITE_JPEG_QUALITY), 25])
    cv2.imwrite(writepath,resized,[int(cv2.IMWRITE_PNG_COMPRESSION),5])


    
def main():
    
    cur_label_dir = ''
    if not os.path.exists(outdir): os.mkdir(outdir)

    laying_dirs = [os.path.join(outdir,'not_laying'),os.path.join(outdir,'laying')]
    nursing_dirs = [os.path.join(laying_dirs[1],'not_nursing'),os.path.join(laying_dirs[1],'nursing')]

    label_dirs = Label_dirs(laying_dirs[0],nursing_dirs[0],nursing_dirs[1])

    for d in laying_dirs:
        if not os.path.exists(d): os.mkdir(d)
    for d in nursing_dirs:
        if not os.path.exists(d): os.mkdir(d)

    is_laying = False
    is_nursing = False
    labeling_started = False
    vid = open_new_video(0)
    frame_i = 0
    cur_time = 0
    with open(label_file, newline='') as tsvfile:
        label_reader = csv.reader(tsvfile,delimiter='\t')
        for row in label_reader:
            if row[0] == "Time" and row[1] == "Media file path":
                labeling_started = True
                continue
            if labeling_started:
                ltime = float(row[0])
                log_string("row time: " + str(ltime) + "  file time: " + get_outfile_name(vid, frame_i)[9:-2])
                cur_time = vid.info.start+frame_i*1/OUTPUT_FPS
                while cur_time < ltime and cur_label_dir:
                    ret,frame,vid,frame_i = read_frame(vid,frame_i)
                    if ret == False:
                        return
                    write_file(frame,vid,frame_i,cur_label_dir)
                    cur_time = vid.info.start+frame_i*1/OUTPUT_FPS
                is_laying,is_nursing = read_label_row(row,is_laying,is_nursing)
                if is_laying==False:
                    cur_label_dir = label_dirs.not_laying
                elif is_nursing == False:
                    cur_label_dir = label_dirs.laying_not_nursing
                else:
                    cur_label_dir = label_dirs.laying_nursing
                    
        
        
if __name__ == "__main__":
    set_opts(args=sys.argv[1:])
    print_elapsed_time()
    main()
    print_elapsed_time('end program')