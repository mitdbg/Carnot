import os
from typing import List, Optional
from datetime import datetime

class FileService:
    """Service for file operations"""
    
    @staticmethod
    def list_directory(path: str) -> List[dict]:
        """List contents of a directory"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Directory not found: {path}")
        
        if not os.path.isdir(path):
            raise NotADirectoryError(f"Path is not a directory: {path}")
        
        items = []
        for entry in os.listdir(path):
            entry_path = os.path.join(path, entry)
            is_dir = os.path.isdir(entry_path)
            stat = os.stat(entry_path)
            
            items.append({
                "name": entry,
                "path": entry_path,
                "is_directory": is_dir,
                "size": stat.st_size if not is_dir else None,
                "modified": datetime.fromtimestamp(stat.st_mtime)
            })
        
        return items
    
    @staticmethod
    def get_file_info(path: str) -> dict:
        """Get information about a file"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        stat = os.stat(path)
        return {
            "name": os.path.basename(path),
            "path": path,
            "is_directory": os.path.isdir(path),
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime)
        }

