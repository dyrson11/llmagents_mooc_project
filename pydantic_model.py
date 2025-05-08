from pydantic import BaseModel, Field
from typing import List, Optional, Union


# Definir el modelo para cada definición de la palabra
class Definicion(BaseModel):
    definicion: str
    pronunciacion: Optional[str] = None 
    PoS: Optional[str] = None
    otras_etiquetas: Optional[List[str]] = None
    sinonimos: Optional[List[str]] = None
    variantes: Optional[List[str]] = None
    
    class Config:
        min_anystr_length = 1
        anystr_strip_whitespace = True
    
    def __str__(self):
        return f"\t- {self.definicion}"

# Modelo principal para una palabra
class Palabra(BaseModel):
    palabra: str  # La palabra en sí
    definiciones: List[Definicion]  # Lista de definiciones de la palabra
    pronunciaciones: Optional[List[str]] = Field(default_factory=list)  # Pronunciaciones comunes (si las hay)
    PoS: Optional[List[str]] = Field(default_factory=list)  # Parte de habla común para todas las definiciones
    sinonimos_comunes: Optional[List[str]] = Field(default_factory=list)  # Sinónimos comunes para todas las definiciones
    
    def __str__(self):
        text = f"\t{self.palabra}: "
        for definicion in self.definiciones:
            text += f"\n\t{definicion}"
        return text


class DefinicionSufijo(BaseModel):
    definicion: str
    descripcion: str
    tipo: str
    naturaleza: str
    
    def __str__(self):
        return f"\t- {self.definicion}: {self.descripcion}"


class Sufijo(BaseModel):
    sufijo: str
    definiciones: List[DefinicionSufijo]
    
    def __str__(self):
        text = f"\t{self.sufijo}: "
        for definicion in self.definiciones:
            text += f"\n\t{definicion}"
        return text


class Posposicion(BaseModel):
    posposicion: str
    definicion: str
    condicion: str
    ejemplo: str
    
    def __str__(self):
        return f"\t{self.posposicion} (posposicion): {self.definicion}"