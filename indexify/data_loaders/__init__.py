from pydantic import BaseModel

from abc import ABC, abstractmethod
from typing import List
import os
import mimetypes
import hashlib

class FileMetadata(BaseModel):
    path: str
    file_size: int
    mime_type: str
    md5_hash: str
    created_at: int
    updated_at: int

    @classmethod
    def from_path(cls, path: str):
        file_size = os.path.getsize(path)
        mime_type = mimetypes.guess_type(path)[0]

        # Compute MD5 hash
        hash_md5 = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        md5_hash = hash_md5.hexdigest()

        created_at = int(os.path.getctime(path))
        updated_at = int(os.path.getmtime(path))

        return cls(
            path=path,
            file_size=file_size,
            mime_type=str(mime_type),
            md5_hash=md5_hash,
            created_at=created_at,
            updated_at=updated_at,
        )

    def read_all_bytes(self) -> bytes:
        with open(self.path, "rb") as f:
            return f.read()


class DataLoader(ABC):
    @abstractmethod
    def load(self) -> List[FileMetadata]:
        pass

    @abstractmethod
    def state(self) -> dict:
        pass

from .local_directory_loader import LocalDirectoryLoader