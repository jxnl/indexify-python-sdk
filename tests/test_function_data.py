import unittest

from indexify.functions_sdk.data_objects import BaseData
from indexify.functions_sdk.output_serializer import CachedOutput, OutputSerializer


class TestBaseData(unittest.TestCase):
    def test_md5_checksum(self):
        data = BaseData(payload="test")
        csum = data.md5_payload_checksum
        self.assertEqual(BaseData(payload="test").md5_payload_checksum, csum)


class TestOutputSerializer(unittest.TestCase):
    def test_serialize_deserialize(self):
        class MyModel(BaseData):
            payload: str

        serializer = OutputSerializer()
        data = CachedOutput(root=[MyModel(payload="test")])
        serialized = serializer.serialize(data)
        deserialized = serializer.deserialize(serialized, MyModel)
        self.assertEqual(data, deserialized)


if __name__ == "__main__":
    unittest.main()
