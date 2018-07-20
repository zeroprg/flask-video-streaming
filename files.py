import os
import glob
import time


# Set the directory you want to start from
def traverse_dir(rootDir=".", wildcard="*" , start = 0 , end = 0, date = None):
    ret = []
    for  file in glob.iglob(rootDir + wildcard,recursive= False):
         ret.append(file)
    if end == 0 or start > end:
        if date == None:
            return sorted(ret, key=os.path.getmtime, reverse=True)[start:]
        else:
            ret = sorted(ret, key=os.path.getmtime, reverse=True)
            right_indx = binary_bisection(compare_dates, date, ret)
            return ret[start:right_indx]  
    else:
        return sorted(ret, key=os.path.getmtime, reverse=True)[start:end]


def delete_file_older_then(path, sec):
    for f in os.listdir(path):
       try:
           if os.stat(os.path.join(path,f)).st_mtime < time.time() - sec:
                os.remove(os.path.join(path, f))
       except OSError: pass

def find_index(list,date):
    return binary_bisection(compare_dates,date,list)
    
""" Compare if the file modification date older then specified 'date'. The goal is to choose all file
    which younger then the specified 'date' """
def compare_dates(file,date):
    # retrieves the stats for the current file as a tuple
    # (mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime)
    # the tuple element mtime at index 8 is the last-modified-date
    stats = os.stat(file)
    # create tuple (year yyyy, month(1-12), day(1-31), hour(0-23), minute(0-59), second(0-59),
    # weekday(0-6, 0 is monday), Julian day(1-366), daylight flag(-1,0 or 1)) from seconds since epoch
    # note:  this tuple can be sorted properly by date and time
    file_lastmod_date = time.localtime(stats[8])
    #iso = time.strftime('%Y-%m-%dT%H:%M:%SZ', lastmod_date)
    """ Return true if file modified after the date (in seconds from 1970)"""
    return date <= file_lastmod_date
    
""" Bisection or Dyhotomy method in generic.  """
def binary_bisection(function,param2,list):
    frm = 0
    to = len(list)
    while frm < to:
        mid = (frm+to)>>1
        if function(list[mid], param2):
            """ Use left side"""
            to = mid
        else:
            """ Use right side"""
            frm = mid+1
    return frm        
