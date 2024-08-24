import inspect
import json
import os
from abc import ABC, abstractmethod
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    get_type_hints,
)

import requests
from pydantic import BaseModel, Field

from .data import BaseData, Content, Feature


class EmbeddingSchema(BaseModel):
    dim: int
    distance: Optional[str] = "cosine"
    database_url: Optional[str] = None


class ExtractorMetadata(BaseModel):
    name: str
    version: str
    description: str
    input_mime_types: List[str]
    system_dependencies: List[str]
    python_dependencies: List[str]
    input_mime_types: List[str]
    embedding_schemas: Dict[str, EmbeddingSchema]
    # Make this a dynamic model since its a json schema
    input_params: Optional[Dict]
    # for backward compatibility
    metadata_schemas: Optional[Dict]


class Extractor(ABC):
    name: str = ""

    version: str = "0.0.0"

    base_image: Optional[str] = None

    system_dependencies: List[str] = []

    python_dependencies: List[str] = []

    description: str = ""

    input_mime_types = ["text/plain"]

    embedding_indexes: Dict[str, EmbeddingSchema] = {}

    @abstractmethod
    def extract(
        self, input: Type[BaseModel], params: Type[BaseModel] = None
    ) -> List[Union[Feature, Type[BaseModel]]]:
        """
        Extracts information from the content. Returns a list of features to add
        to the content.
        It can also return a list of Content objects, which will be added to storage
        and any extraction policies defined will be applied to them.
        """
        pass

    @classmethod
    def sample_input(cls) -> Tuple[Content, Type[BaseModel]]:
        pass

    def _download_file(self, url, filename):
        if os.path.exists(filename):
            # file exists skip
            return
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()  # Raises an HTTPError if the response status code is 4XX/5XX
                with open(filename, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        except requests.exceptions.RequestException as e:
            print(f"Error downloading the file: {e}")

    def sample_mp3(self, features: List[Feature] = []) -> Content:
        file_name = "sample.mp3"
        self._download_file(
            "https://extractor-files.diptanu-6d5.workers.dev/sample-000009.mp3",
            file_name,
        )
        f = open(file_name, "rb")
        return Content(content_type="audio/mpeg", data=f.read(), features=features)

    def sample_mp4(self, features: List[Feature] = []) -> Content:
        file_name = "sample.mp4"
        self._download_file(
            "https://extractor-files.diptanu-6d5.workers.dev/sample.mp4",
            file_name,
        )
        f = open(file_name, "rb")
        return Content(content_type="video/mp4", data=f.read(), features=features)

    def sample_jpg(self, features: List[Feature] = []) -> Content:
        file_name = "sample.jpg"
        self._download_file(
            "https://extractor-files.diptanu-6d5.workers.dev/people-standing.jpg",
            file_name,
        )
        f = open(file_name, "rb")
        return Content(content_type="image/jpg", data=f.read(), features=features)

    def sample_invoice_jpg(self, features: List[Feature] = []) -> Content:
        file_name = "sample.jpg"
        self._download_file(
            "https://extractor-files.diptanu-6d5.workers.dev/invoice-example.jpg",
            file_name,
        )
        f = open(file_name, "rb")
        return Content(content_type="image/jpg", data=f.read(), features=features)

    def sample_invoice_pdf(self, features: List[Feature] = []) -> Content:
        file_name = "sample.pdf"
        self._download_file(
            "https://extractor-files.diptanu-6d5.workers.dev/invoice-example.pdf",
            file_name,
        )
        f = open(file_name, "rb")
        return Content(content_type="application/pdf", data=f.read(), features=features)

    def sample_image_based_pdf(self, features: List[Feature] = []) -> Content:
        file_name = "sample.pdf"
        self._download_file(
            "https://extractor-files.diptanu-6d5.workers.dev/image-based.pdf",
            file_name,
        )
        f = open(file_name, "rb")
        return Content(content_type="application/pdf", data=f.read(), features=features)

    def sample_scientific_pdf(self, features: List[Feature] = []) -> Content:
        file_name = "sample.pdf"
        self._download_file(
            "https://extractor-files.diptanu-6d5.workers.dev/scientific-paper-example.pdf",
            file_name,
        )
        f = open(file_name, "rb")
        return Content(content_type="application/pdf", data=f.read(), features=features)

    def sample_presentation(self, features: List[Feature] = []) -> Content:
        file_name = "test.pptx"
        self._download_file(
            "https://raw.githubusercontent.com/tensorlakeai/indexify/main/docs/docs/files/test.pptx",
            file_name,
        )
        f = open(file_name, "rb")
        return Content(
            content_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            data=f.read(),
            features=features,
        )

    def sample_text(self, features: List[Feature] = []) -> Content:
        article = """New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York. A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband. Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other. In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage. Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the 2010 marriage license application, according to court documents. Prosecutors said the marriages were part of an immigration scam. On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further. After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002. All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say. Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages. Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted. The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali. Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force. If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18."""
        return Content(content_type="text/plain", data=article, features=features)

    def sample_html(self, features: List[Feature] = []) -> Content:
        file_name = "sample.html"
        self._download_file(
            "https://extractor-files.diptanu-6d5.workers.dev/sample.html",
            file_name,
        )
        f = open(file_name, "rb")
        return Content(content_type="text/html", data=f.read(), features=features)


def extractor(
    name: Optional[str] = None,
    description: Optional[str] = "",
    version: Optional[str] = "",
    python_dependencies: Optional[List[str]] = None,
    system_dependencies: Optional[List[str]] = None,
    input_mime_types: Optional[List[str]] = None,
    embedding_indexes: Optional[Dict[str, EmbeddingSchema]] = None,
    sample_content: Optional[Callable] = None,
):
    args = locals()
    del args["sample_content"]

    def construct(fn):
        def wrapper():
            description = fn.__doc__ or args.get("description", "")

            if not args.get("name"):
                args[
                    "name"
                ] = f"{inspect.getmodule(inspect.stack()[1][0]).__name__}:{fn.__name__}"

            class DecoratedFn(Extractor):
                @classmethod
                def extract(cls, input: Type[BaseData], params: Type[BaseModel] = None) -> List[Union[Type[BaseModel], Type[Feature]]]:  # type: ignore
                    # TODO we can force all the functions to take in a parms object
                    # or check if someone adds a params
                    if params is None:
                        return fn(input)
                    else:
                        return fn(input, params)

                def sample_input(self) -> Content:
                    return sample_content() if sample_content else self.sample_text()

            for key, val in args.items():
                setattr(DecoratedFn, key, val)
            DecoratedFn.description = description

            return DecoratedFn

        wrapper._extractor_name = fn.__name__
        wrapper.name = fn.__name__

        return wrapper

    return construct
