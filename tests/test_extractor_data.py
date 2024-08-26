from indexify.extractor_sdk.data import BaseData
import unittest

class TestBaseData(unittest.TestCase):
    def test_md5_checksum(self):
        data = BaseData(payload="test")
        csum = data.md5_payload_checksum
        self.assertEqual(BaseData(payload="test").md5_payload_checksum, csum)


if __name__ == "__main__":
    unittest.main()

