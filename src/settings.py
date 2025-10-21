import os

from pydantic import BaseModel

DIR = os.getcwd()

class CollectorSettings(BaseModel):
    pass
