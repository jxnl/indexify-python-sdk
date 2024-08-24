from dataclasses import asdict, dataclass
from typing import List, Optional


@dataclass
class ExtractionPolicy:
    extractor: str
    name: str
    content_source: str
    input_params: Optional[dict] = None
    id: Optional[str] = None
    labels_eq: Optional[str] = None

    def __repr__(self) -> str:
        return f"ExtractionPolicy(name={self.name} extractor={self.extractor})"

    def __str__(self) -> str:
        return self.__repr__()

    def to_dict(self) -> dict:
        filtered_dict = {k: v for k, v in asdict(self).items() if v is not None}
        return filtered_dict

    @classmethod
    def from_dict(cls, json: dict):
        if "filters_eq" in json:
            json["labels_eq"] = json.pop("filters_eq")
        json["id"] = json.get("id", None)
        return ExtractionPolicy(**json)


@dataclass
class ExtractionGraph:
    id: str
    name: str
    description: str
    extraction_policies: List[ExtractionPolicy]

    @classmethod
    def from_dict(cls, json: dict):
        json["id"] = json.get("id", None)
        json["description"] = json.get("description", None)
        if "namespace" in json.keys():
            json.pop("namespace")
        return ExtractionGraph(**json)

    @staticmethod
    def from_yaml(spec: str):
        import yaml

        return ExtractionGraph.from_dict(yaml.load(spec, Loader=yaml.FullLoader))

    @staticmethod
    def from_yaml_file(path: str):
        with open(path, "r") as f:
            return ExtractionGraph.from_yaml(f.read())

    def to_dict(self) -> dict:
        filtered_dict = {k: v for k, v in asdict(self).items() if v is not None}
        return filtered_dict


class ExtractionGraphBuilder:
    def __init__(self, name: str):
        self.name = name
        self.extraction_policies = []

    def policy(self, policy: ExtractionPolicy) -> "ExtractionGraphBuilder":
        self.extraction_policies.append(policy)
        return self

    def build(self):
        return ExtractionGraph(
            id=self.id, name=self.name, extraction_policies=self.extraction_policies
        )
