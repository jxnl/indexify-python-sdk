import unittest
from typing import List

from indexify.extractor_sdk.data import BaseData
from indexify.extractor_sdk.extractor import ExtractorWrapper, extractor
from indexify.extractor_sdk.local_cache import CacheAwareExtractorWrapper


class TestExtractorWrapper(unittest.TestCase):
    def test_basic_features(self):
        @extractor()
        def extractor_a(url: str) -> str:
            """
            Random description of extractor_a
            """
            return "hello"

        extractor_wrapper = ExtractorWrapper(extractor_a)
        result = extractor_wrapper.extract(BaseData.from_data(url="foo"))
        self.assertEqual(result[0].payload, "hello")

    def test_get_output_model(self):
        @extractor()
        def extractor_b(url: str) -> str:
            """
            Random description of extractor_b
            """
            return "hello"

        extractor_wrapper = ExtractorWrapper(extractor_b)
        result = extractor_wrapper.get_output_model()
        self.assertEqual(result, str)

    def test_list_output_model(self):
        @extractor()
        def extractor_b(url: str) -> List[str]:
            """
            Random description of extractor_b
            """
            return ["hello", "world"]

        extractor_wrapper = ExtractorWrapper(extractor_b)
        result = extractor_wrapper.get_output_model()
        self.assertEqual(result, List[str])


class TestCacheAwareExtractorWrapper(unittest.TestCase):
    def test_cache_aware_extractor_wrapper(self):
        @extractor()
        def extractor_a(url: str) -> str:
            """
            Random description of extractor_a
            """
            return "hello"

        extractor_wrapper = ExtractorWrapper(extractor_a)
        cache_aware_extractor_wrapper = CacheAwareExtractorWrapper(
            "extractor_cache", "test_graph", extractor_wrapper
        )
        result = cache_aware_extractor_wrapper.extract(
            "extractor_a", BaseData.from_data(url="foo")
        )
        self.assertEqual(result[0].payload, "hello")

    def test_list_output_model_cache(self):
        @extractor()
        def extractor_b(url: str) -> List[str]:
            """
            Random description of extractor_b
            """
            return ["hello", "world"]

        extractor_wrapper = ExtractorWrapper(extractor_b)
        cache_aware_extractor_wrapper = CacheAwareExtractorWrapper(
            "extractor_cache", "test_graph", extractor_wrapper
        )
        result = cache_aware_extractor_wrapper.extract(
            "extractor_b", BaseData.from_data(url="foo")
        )
        self.assertEqual(result[0].payload, "hello")
        self.assertEqual(result[1].payload, "world")

    def test_cache_aware_extractor_wrapper_with_bytes(self):
        from pydantic import BaseModel

        class TestModel(BaseModel):
            payload: bytes
            metadata: str
            some_value: int

        @extractor()
        def extractor_x(url: str) -> TestModel:
            """
            Random description of extractor_c
            """
            return TestModel(payload=b"hello", metadata="world", some_value=1)

        extractor_wrapper = ExtractorWrapper(extractor_x)
        cache_aware_extractor_wrapper = CacheAwareExtractorWrapper(
            "extractor_cache", "test_graph", extractor_wrapper
        )
        result = cache_aware_extractor_wrapper.extract(
            "extractor_x", BaseData.from_data(url="foo")
        )
        self.assertEqual(result[0].payload.payload, b"hello")
        self.assertEqual(result[0].payload.metadata, "world")
        self.assertEqual(result[0].payload.some_value, 1)


if __name__ == "__main__":
    unittest.main()
