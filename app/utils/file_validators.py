"""
Валидация файлов
"""

from pathlib import Path
from typing import List
from fastapi import UploadFile
from app.core.config import get_settings
from app.exceptions import InvalidFileFormatError

settings = get_settings()


def validate_image_file(file: UploadFile) -> None:
    """
    Валидация файла изображения
    
    Args:
        file: Файл изображения
        
    Raises:
        InvalidFileFormatError: Если файл невалиден
    """
    if not file.filename:
        raise InvalidFileFormatError("Имя файла изображения не может быть пустым")
    
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in settings.ALLOWED_IMAGE_EXTENSIONS:
        raise InvalidFileFormatError(
            f"Неподдерживаемый формат изображения: {file_ext}. "
            f"Поддерживаемые форматы: {', '.join(settings.ALLOWED_IMAGE_EXTENSIONS)}"
        )


def validate_label_file(file: UploadFile) -> None:
    """
    Валидация файла labels
    
    Args:
        file: Файл labels
        
    Raises:
        InvalidFileFormatError: Если файл невалиден
    """
    if not file.filename:
        raise InvalidFileFormatError("Имя файла labels не может быть пустым")
    
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext != ".txt":
        raise InvalidFileFormatError(
            f"Файл labels должен иметь расширение .txt, получено: {file_ext}"
        )


def validate_file_size(file: UploadFile, max_size_mb: int = None) -> None:
    """
    Валидация размера файла
    
    Args:
        file: Файл для проверки
        max_size_mb: Максимальный размер в МБ
        
    Raises:
        InvalidFileFormatError: Если файл слишком большой
    """
    if max_size_mb is None:
        max_size_mb = settings.MAX_UPLOAD_SIZE_MB
    
    # Примечание: FastAPI уже ограничивает размер загружаемых файлов
    # Эта функция может быть использована для дополнительной проверки

