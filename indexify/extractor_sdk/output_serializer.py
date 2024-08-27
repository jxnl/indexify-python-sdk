import json
import os
import uuid
from typing import List, Type, Union

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
        def serialize_payload(payload):
            if isinstance(payload, dict):
                serialized_dict = {}
                for field_name, field_value in payload.items():
                    if isinstance(field_value, bytes):
                        serialized_dict[field_name] = self.bytes_serializer.serialize(
                            field_value
                        )
                    elif isinstance(field_value, (dict, list)):
                        serialized_dict[field_name] = serialize_payload(field_value)
                    else:
                        serialized_dict[field_name] = field_value
                return serialized_dict
            elif isinstance(payload, list):
                serialized_list = []
                for item in payload:
                    if isinstance(item, bytes):
                        serialized_list.append(self.bytes_serializer.serialize(item))
                    elif isinstance(item, (dict, list)):
                        serialized_list.append(serialize_payload(item))
                    else:
                        serialized_list.append(item)
                return serialized_list
            else:
                if isinstance(payload, bytes):
                    return self.bytes_serializer.serialize(payload)
                else:
                    return payload

        dict_to_serialize = []
        for data in outputs.model_dump():
            base_data_dict = {}
            payload = data.pop("payload")
            base_data_dict.update(data)
            base_data_dict["payload"] = serialize_payload(payload)
            dict_to_serialize.append(base_data_dict)

        return json.dumps(dict_to_serialize)

    def deserialize(self, json_str: str, model_class: Type[BaseModel]) -> BaseModel:
        def deserialize_payload(payload, field_type):
            if isinstance(payload, dict):
                deserialized_dict = {}
                for field_name, field_value in payload.items():
                    sub_field_type = field_type.model_fields[field_name].annotation
                    if sub_field_type is bytes:
                        deserialized_dict[
                            field_name
                        ] = self.bytes_serializer.deserialize(field_value)
                    elif hasattr(sub_field_type, "__origin__"):
                        if sub_field_type.__origin__ is Union:
                            inner_types = sub_field_type.__args__
                            for inner_type in inner_types:
                                if issubclass(inner_type, BaseModel):
                                    deserialized_dict[field_name] = deserialize_payload(
                                        field_value, inner_type
                                    )
                                    break
                        elif issubclass(sub_field_type.__origin__, BaseModel):
                            deserialized_dict[field_name] = deserialize_payload(
                                field_value, sub_field_type
                            )
                        elif sub_field_type.__origin__ is list:
                            deserialized_dict[field_name] = deserialize_payload(
                                field_value, sub_field_type
                            )
                        deserialized_dict[field_name] = deserialize_payload(
                            field_value, sub_field_type
                        )
                    else:
                        deserialized_dict[field_name] = field_value
                        deserialized_dict[field_name] = field_value

                return deserialized_dict
            elif isinstance(payload, list):
                deserialized_list = []
                for item in payload:
                    if field_type.__args__[0] is bytes:
                        deserialized_list.append(
                            self.bytes_serializer.deserialize(item)
                        )
                    elif issubclass(field_type.__args__[0], BaseModel):
                        deserialized_list.append(
                            deserialize_payload(item, field_type.__args__[0])
                        )
                    else:
                        deserialized_list.append(item)
                return deserialized_list
            else:
                if field_type is bytes:
                    return self.bytes_serializer.deserialize(payload)
                else:
                    return payload

        data = json.loads(json_str)
        outputs = []
        for base_data in data:
            payload_field_type = model_class.model_fields["payload"].annotation
            base_data["payload"] = deserialize_payload(
                base_data["payload"], payload_field_type
            )
            outputs.append(model_class(**base_data))
        return outputs
