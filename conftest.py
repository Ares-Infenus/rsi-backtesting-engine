# conftest.py — Raíz del proyecto
# Este archivo le indica a pytest que la raíz del proyecto es el directorio actual,
# lo que permite que los imports `from src.xxx import ...` funcionen correctamente.
import sys
import os

# Agrega la raíz del proyecto al sys.path automáticamente
sys.path.insert(0, os.path.dirname(__file__))
