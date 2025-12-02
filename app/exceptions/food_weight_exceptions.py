"""
Кастомные исключения для сервиса определения веса продуктов
"""


class FoodWeightException(Exception):
    """Базовое исключение для сервиса"""
    pass


class ImageProcessingError(FoodWeightException):
    """Ошибка обработки изображения"""
    pass


class InvalidFileFormatError(FoodWeightException):
    """Неподдерживаемый формат файла"""
    pass


class LabelParsingError(FoodWeightException):
    """Ошибка парсинга YOLO labels"""
    pass


class CalibrationError(FoodWeightException):
    """Ошибка калибровки масштаба"""
    pass


class VolumeCalculationError(FoodWeightException):
    """Ошибка расчета объема"""
    pass

