import unittest
import timer
from ObjCountByTimer import ObjCountByTimer

class TestObjCountByTimer(unittest.TestCase):

    def test_description(self):

        obj = ObjCountByTimer(10, [1,2,5,10])
        obj.add("Privet Lunatikam!!!")
        obj.add("Privet!!!")
        obj.add("Pri!!!")
        time.sleep(80)
        self.assertEqual(obj.get(1), 3)
        print('obj.get(0):{} '.format(obj.get(1)))

test = TestObjCountByTimer()
test.test_description
