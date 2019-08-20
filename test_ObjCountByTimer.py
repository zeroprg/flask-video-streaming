import unittest
import time
from objCountByTimer import ObjCountByTimer

class TestObjCountByTimer(unittest.TestCase):

    def test_description(self):

        obj = ObjCountByTimer(1,65, (5,10,60))
        obj.add("Privet Lunatikam!!!")
        obj.add("Privet!!!")
        obj.add("Pri!!!")
        time.sleep(5)
        # subtest 1
        print('obj.store: {} obj.sliced_result {}'.format(obj.store, obj.counted))
        print('How many different obj.counted[0] after 5 sec: {}'.format(obj.counted[0]))
        self.assertEqual(obj.counted[0], 3)
        time.sleep(5.0)
        
        # subtest 2
        print('obj.store: {} obj.sliced_result {}'.format(obj.store, obj.counted))
        print('How many different obj.counted[1] after 10  sec: {}'.format(obj.counted[1]))
        self.assertEqual(obj.counted[1], 3)
        time.sleep(90.0)

        # subtest 3
        print('obj.store: {} obj.sliced_result {}'.format(obj.store, obj.counted))
        print('How many different obj.counted[2] after >60 (100sec)  sec: {}'.format(obj.counted[2]))
        self.assertEqual(obj.counted[2], 3)
        
        obj.stop()


test = TestObjCountByTimer()
test.test_description
