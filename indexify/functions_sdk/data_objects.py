import hashlib
import pickle
from typing import Any, Dict, Optional

from pydantic import BaseModel, Json, create_model


class BaseData(BaseModel):
    content_id: Optional[str] = None
    payload: Optional[Any] = None
    md5_payload_checksum: Optional[str] = None

    def model_post_init(self, __context):
        hash = hashlib.md5()
        for k, v in self.model_dump().items():
            if k == "md5_payload_checksum":
                continue
            hash.update(k.encode())
            hash.update(pickle.dumps(v))
        self.md5_payload_checksum = hash.hexdigest()

    @staticmethod
    def from_data(**kwargs) -> "BaseData":
        fields = {key: (type(value), ...) for key, value in kwargs.items()}
        DynamicModel = create_model("DynamicModel", **fields)
        return BaseData(payload=DynamicModel(**kwargs))


class File(BaseModel):
    data: bytes
    mime_type: str
    metadata: Optional[Dict[str, Json]] = None
