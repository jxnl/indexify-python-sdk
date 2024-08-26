import json
import os
import uuid
from typing import List, Type

from pydantic import BaseModel, RootModel

from .data import BaseData

CachedOutput = RootModel[List[BaseData]]


class BytesFieldSerializer:
    def __init__(self, data_dir="./data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def serialize(self, value: bytes) -> str:
        file_name = str(uuid.uuid4())
        file_path = os.path.join(self.data_dir, file_name)
        with open(file_path, "wb") as f:
            f.write(value)
        return file_name

    def deserialize(self, file_name: str) -> bytes:
        file_path = os.path.join(self.data_dir, file_name)
        with open(file_path, "rb") as f:
            return f.read()


class PayloadSerializer:
    def __init__(self, data_dir="./data"):
        self.bytes_serializer = BytesFieldSerializer(data_dir)

    def serialize(self, outputs: CachedOutput) -> str:
        dict_to_serialize = []
        for data in outputs.model_dump():
            base_data_dict = {}
            payload = data.pop("payload")
            base_data_dict.update(data)
            base_data_dict["payload"] = {}
            if isinstance(payload, dict):
                for field_name, field_value in payload.items():
                    if isinstance(field_value, bytes):
                        base_data_dict["payload"][
                            field_name
                        ] = self.bytes_serializer.serialize(field_value)
                    else:
                        base_data_dict["payload"][field_name] = field_value
            elif isinstance(payload, list):
                for i, items in enumerate(payload):
                    if isinstance(items, bytes):
                        base_data_dict["payload"][i] = self.bytes_serializer.serialize(
                            items
                        )
                    else:
                        base_data_dict["payload"][i] = items
            else:
                if isinstance(payload, bytes):
                    base_data_dict["payload"] = self.bytes_serializer.serialize(payload)
                else:
                    base_data_dict["payload"] = payload
            dict_to_serialize.append(base_data_dict)

        return json.dumps(dict_to_serialize)

    def deserialize(self, json_str: str, model_class: Type[BaseModel]) -> BaseModel:
        data = json.loads(json_str)
        outputs = []
        for base_data in data:
            if isinstance(base_data["payload"], dict):
                for field_name, field_value in base_data["payload"].items():
                    field_type = (
                        model_class.model_fields["payload"]
                        .annotation.model_fields[field_name]
                        .annotation
                    )
                    if field_type is bytes:
                        base_data["payload"][
                            field_name
                        ] = self.bytes_serializer.deserialize(field_value)
            elif isinstance(base_data["payload"], list):
                for i, items in enumerate(base_data["payload"]):
                    if isinstance(items, bytes):
                        base_data["payload"][i] = self.bytes_serializer.deserialize(
                            items
                        )
            else:
                if isinstance(base_data["payload"], bytes):
                    base_data["payload"] = self.bytes_serializer.deserialize(
                        base_data["payload"]
                    )
            outputs.append(model_class(**base_data))
        return outputs
