""" 
    Functions for logging
"""

import numpy as np
import time
import os
import sys
import datetime
import us_timezone


def log_out(FOUT=sys.stdout,*args):
    out_str = ' '.join(map(str, args))
    FOUT.write(str(out_str)+'\n')
    FOUT.flush()
    if not FOUT == sys.stdout:
        print(out_str)
        
def copy_file(LOG_DIR,filename):
    if os.path.exists(filename):
        os.system('cp '+str(filename)+' %s' % (LOG_DIR))
        
def make_log_dir_pointnets(FLAGS):
    log_dir_base = FLAGS.log_dir
    if not os.path.exists(log_dir_base): os.mkdir(log_dir_base)
    date_str = self.start_dt.strftime('%Y-%m-%d-%H-%M')
    cur_log_folder = date_str+'_grid' \
    + str(FLAGS.grid_size).zfill(2) \
    + '_point'+str(FLAGS.num_point).zfill(3) \
    + '_epoch'+str(FLAGS.max_epoch).zfill(3)
    if FLAGS.xyzonly:
        cur_log_folder = cur_log_folder + '_xyzonly'
    self.LOG_DIR = os.path.join(FLAGS.log_dir,cur_log_folder)
    
def print_time(FOUT,start_time,section_name=''):
    total_secs = int(time.time()-start_time)
    #days = total_secs//(60*60*24)
    #leftover_secs = total_secs % (60*60*24)
    hours = total_secs//(60*60)
    leftover_secs = total_secs % (60*60)
    minutes = leftover_secs//60
    seconds = leftover_secs % 60
    intro = '---- ' + section_name + ' run time: '
    log_out(FOUT,intro + '%d hours, %d minutes, %d seconds ----' \
                   % (hours, minutes, seconds))
    
    
    
def printTable (FOUT, tbl, header=None,col_names=[], borderHoriz = '-', borderVert = '|', 
                borderCross = '+',start_borderHoriz='=',start_borderVert='||',left_edges=True):
    def make_str(x):
        if isinstance(x, int):
            return '{:d}'.format(x)
        if isinstance(x, float):
            if x < 1:
                return '{:.2%}'.format(x)
            return '{:.2f}'.format(x)
        return str(x)

    if left_edges:
        borders = {'edgeH':borderHoriz, 'edge_startH':start_borderHoriz,
                  'startH':start_borderHoriz, 'H':borderHoriz,
                  'edgeV':borderVert, 'edge_startV':start_borderVert,
                  'startV':start_borderVert, 'V':borderVert,
                  'edge_cross':borderCross, 'cross':borderCross}
    else:
        borders = {'edgeH':' ','edge_startH':' ','startH':start_borderHoriz,'H':borderHoriz,
                  'edgeV':' ','edge_startV':' ','startV':start_borderVert,'V':borderVert,
                  'edge_cross':' ','cross':borderCross}        

    isCN = bool(col_names)

    cols = [list(map(make_str,x)) for x in zip(*tbl)]
    lengths = [max(map(len, map(str, col))) for col in cols]

    if isCN: 
        cols = [col_names] + cols
        if header:
             header = [' '] + header
    cl = max(list(map(len,map(str,col_names))) + [0])
    tbl = [list(x) for x in zip(*cols)]    
    start_f = borders['edgeV'] \
    + (' {:>%d} '% cl)*isCN \
    + borders['edge_startV']*isCN \
    + borders['edgeV'].join(' {:>%d} ' % l for l in lengths) \
    + borders['edgeV']

    start_s = borders['edge_cross'] \
    + borders['edge_startH']*(cl+2)*isCN \
    + (borders['cross']*len(borders['startV'])*isCN) \
    + borders['cross'].join(borders['startH'] * (l+2) for l in lengths) \
    + borders['cross']

    if header and not isCN: 
        borders['edgeV'] = borderVert
        borders['edge_cross'] = borderCross

    f = borders['edgeV'] \
    + (' {:>%d} '% cl)*isCN \
    + borders['startV']*isCN \
    + borders['V'].join(' {:>%d} ' % l for l in lengths) \
    + borders['V']

    s = borders['edge_cross'] \
    + borders['edgeH']*(cl+2)*isCN \
    + (borders['cross']*len(borders['startV'])*isCN) \
    + borders['cross'].join(borders['H'] * (l+2) for l in lengths) \
    + borders['cross']

    if left_edges:
        if header:
            log_out(FOUT,s)
        else:
            log_out(FOUT,start_s)
    if header:
        log_out(FOUT,start_f.format(*header))
        log_out(FOUT,start_s)
    for row in tbl:
        log_out(FOUT,f.format(*row))
        log_out(FOUT,s)

class Log:
    def __init__(self, save_folder='',log_file='log.txt',timezone=us_timezone.Pacific,debug=False):
        self.start_time = time.time()
        self.start_dt = datetime.datetime.now(timezone)
        if not os.path.exists(save_folder): os.mkdir(save_folder)
        if debug:
            self.LOG_DIR = os.path.join(save_folder, 'debug')
        else:
            time_string = self.start_dt.strftime('%Y-%m-%d_%H-%M')
            self.LOG_DIR = os.path.join(save_folder, time_string)
        if not os.path.exists(self.LOG_DIR): os.mkdir(self.LOG_DIR)
        self.LOG_FOUT = open(os.path.join(self.LOG_DIR, log_file), 'w')
        
    def print_time(self,start_time=None,section_name=''):
        if not start_time:
            start_time = self.start_time
        print_time(self.LOG_FOUT,start_time,section_name=section_name)
        
    def copy_file(self,filename):
        copy_file(self.LOG_DIR,filename)

    def out(self,*args):
        log_out(self.LOG_FOUT,*args)
     
    def close(self):
        print_time(self.LOG_FOUT,self.start_time)
        self.LOG_FOUT.close()
        
    
    def printTable (self, tbl, header=None,col_names=[], borderHoriz = '-', borderVert = '|', 
                    borderCross = '+',start_borderHoriz='=',start_borderVert='||',left_edges=True):
        printTable(self.LOG_FOUT,tbl,header,col_names,borderHoriz,borderVert,borderCross, \
                   start_borderHoriz,start_borderVert,left_edges)