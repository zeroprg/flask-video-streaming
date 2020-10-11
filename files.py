import os
import glob
import time
import json
import re

# Set the directory you want to start from
def traverse_dir(rootDir=".", reverse = True, wildcard="*" , start = 0 , end = 0, date = None):
    ret = []
    for  file in glob.iglob(rootDir + wildcard,recursive= False):
         ret.append(file)
    if end == 0 or start > end:
        if date == None:
            return sorted(ret, key=os.path.getmtime, reverse=reverse)[start:]
        else:
            ret = sorted(ret, key=os.path.getmtime, reverse=reverse)
            right_indx = binary_bisection(compare_dates, date, ret)
            return ret[start:right_indx]  
    else:
        return sorted(ret, key=os.path.getmtime, reverse=reverse)[start:end]


def delete_file_older_then(path, sec):
    for f in os.listdir(path):
       try:
           if os.stat(os.path.join(path,f)).st_mtime < time.time()*1000 - sec:
                os.remove(os.path.join(path, f))
       except OSError: pass

def find_index(list,date):
    #print("find_index, date:", date)
    i = 0
    for el in list:
        if compare_dates(el,date): return i
        i+=1  
    return len(list)-1    
    #return binary_bisection(compare_dates,date,list)
    
""" Compare if the file modification date older then specified 'date'. The goal is to choose all file
    which younger then the specified 'date' """
def compare_dates(file,date):
    # retrieves the stats for the current file as a tuple
    # (mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime)
    # the tuple element mtime at index 8 is the last-modified-date
    match = re.search(r'.*(?:\D|^)(\d+)', file)
    print(match)
    if not match:  
        raise Exception("Invalid file name, it must be in format 12345.* where the number is time in seconds from epoch (January 1st, 1970)!: " + file)
    file_lastmod_date = match.group(1)
    """ Return true if file modified after the date (in seconds from 1970)"""
    #print("file_lastmod_date:",  file_lastmod_date)
    return float(file_lastmod_date) < float(date)
    
""" Bisection or Dyhotomy method in generic.  """
def binary_bisection(function,date,list):
    frm = 0
    old_mid = -1
    to = len(list)
    print(to)
    while frm < to:
        mid = (to - frm)>>1 
        if mid == old_mid: break
        old_mid = mid
        print("list[mid]:",mid, list[mid])
        #Compare if the file modification date older then specified 'date'.
        # list[mid] < date
        if  function(list[mid], date):
            """ Use left side"""
            frm = mid
            
        else:
            """ Use right side"""
            to = mid - 1
        print("frm,to:",frm,to)
    return frm        

if (__name__ == '__main__'):
    PARAMS_FOLDER = "static/params/" 
    params_files = traverse_dir(PARAMS_FOLDER, False)
    print("Test #1: Populate list of JSON files with form folder")
    print("params_files:" + json.dumps(params_files))
    if len(params_files)>0: print("Test #1 : !!!!!!!!!!!! Successed !!!!!!!!!!!!!!")
    print("Test #2: Compare modification of first file in list with provided time")
    test2Result = compare_dates(params_files[0], 1534367187000) #time.time()*1000)
    print(test2Result)
    if test2Result: print("Test #2 : !!!!!!!!!!!! Successed !!!!!!!!!!!!!!")
    print("Test #3")
    index = find_index(params_files, 1534464194)
    if index>0: print("Test #3 : !!!!!!!!!!!! Successed !!!!!!!!!!!!!!")
    
    
    