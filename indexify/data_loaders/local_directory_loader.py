from . import DataLoader, FileMetadata
from typing import List, Optional
import os


class LocalDirectoryLoader(DataLoader):
    def __init__(self, directory: str, file_extensions: Optional[List[str]] = None):
        self.directory = directory
        self.file_extensions = file_extensions
        self.processed_files = set()

    def load(self) -> List[FileMetadata]:
        file_metadata_list = []
        for root, _, files in os.walk(self.directory):
            for file in files:
                if self.file_extensions is None or any(
                    file.endswith(ext) for ext in self.file_extensions
                ):
                    file_path = os.path.join(root, file)
                    if file_path not in self.processed_files:
                        file_metadata_list.append(FileMetadata.from_path(file_path))
                        self.processed_files.add(file_path)

        return file_metadata_list

    def state(self) -> dict:
        return {"processed_files": list(self.processed_files)}
