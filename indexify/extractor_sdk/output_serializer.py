from pydantic import BaseModel 
import json
import os
from typing import Type
import uuid

class BytesFieldSerializer:
    def __init__(self, data_dir='./data'):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def serialize(self, value: bytes) -> str:
        file_name = str(uuid.uuid4())
        file_path = os.path.join(self.data_dir, file_name)
        with open(file_path, 'wb') as f:
            f.write(value)
        return file_name

    def deserialize(self, file_name: str) -> bytes:
        file_path = os.path.join(self.data_dir, file_name)
        with open(file_path, 'rb') as f:
            return f.read()

class PayloadSerializer:
    def __init__(self, data_dir='./data'):
        self.bytes_serializer = BytesFieldSerializer(data_dir)

    def serialize(self, model: BaseModel) -> str:
        data = model.model_dump()
        print(data)
        for field_name, field_value in data.items():
            if isinstance(field_value, bytes):
                data[field_name] = self.bytes_serializer.serialize(field_value)
        return json.dumps(data)

    def deserialize(self, json_str: str, model_class: Type[BaseModel]) -> BaseModel:
        data = json.loads(json_str)
        for field_name, field_value in data.items():
            field_type = model_class.model_fields[field_name].annotation
            if field_type is bytes:
                data[field_name] = self.bytes_serializer.deserialize(field_value)
        return model_class(**data)
