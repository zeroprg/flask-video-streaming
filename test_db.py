import unittest
import time
import datetime
import numpy as np
import random as r

import db

DB_LOCATION = "framedata.db"
now = datetime.datetime.now()


class TestDB(unittest.TestCase):
    #  test 1
    def test_connection(self):
        conn = db.create_connection(DB_LOCATION)

    #  test 2
    def test_insert(self):
        conn = db.create_connection(DB_LOCATION)
        day = "{date:%Y-%m-%d}".format(date=now)
        print("Current day: " + day)
        time = "{time:%Y-%m-%d %H:%M:%S}".format(time=now)
        print("Current time: " + time)
        hashcode = 12345467890 * r.randint(1, 5000)
        numpy_array = np.random.rand(100,100)
        print("numpy array: {} ".format(numpy_array) )
        db.insert_frame(conn, hashcode=hashcode, date=day, time=time, type="car" , numpy_array=numpy_array, x_dim=100, y_dim=100)



    def test_select_all(self):
        conn = db.create_connection(DB_LOCATION)
        rows = db.select_all_stats(conn)
        self.assertEqual( len(rows)>0, True);

    def test_select_frame_by_time(self):
        conn = db.create_connection(DB_LOCATION)
        # get time 1 second before creation
        time1 = now - datetime.timedelta(seconds = 1)
        # get time 1 second after creation
        time2 = now + datetime.timedelta(seconds = 1)
        ls = db.select_frame_by_time(conn, time1, time2)
        self.assertEqual(len(ls), 1)

    def test_time_in_epoch_format(self):
        time.sleep(1.0)
        print("Test time in epoch format")
        conn = db.create_connection(DB_LOCATION)
        day = "{date:%Y-%m-%d}".format(date=now)
        print("Current day: " + day)
        time1 = time.time()
        print("Current time in epoch seconds: {}".format(time))
        hashcode = 12345467890 * r.randint(1, 5000)
        numpy_array = np.random.rand(100,100)
        print("numpy array: {} ".format(numpy_array) )
        db.insert_frame(conn, hashcode=hashcode, date=day, time=time1, type="car" , numpy_array=numpy_array, x_dim=100, y_dim=100)
        conn = db.create_connection(DB_LOCATION)
        # get time 1 second before creation
        time1 -= 1
        # get time 1 second after creation
        time2 = time1 + 1
        ls = db.select_frame_by_time(conn, time1, time2)
        self.assertEqual(len(ls), 1)
        print("Current time from DB:".format(ls[0][2]))

    def test_delete_by_time(self):
        conn = db.create_connection(DB_LOCATION)
        time.sleep(2)
        print("Test delete records n minutes before")
        predicate = "'-1 seconds'"  
        db.delete_frames_later_then(conn,predicate)
        rows = db.select_all_stats(conn)
        self.assertEqual(len(rows),0);



test = TestDB()
test.test_connection
test.test_insert
test.test_select_all
test.test_select_frame_by_time
test.test_time_in_epoch_format
test.test_delete_by_time
 
