"""
Python Character Mapping Codec for CP858, modified from cp850.
"""

import codecs
from _typeshed import ReadableBuffer

class Codec(codecs.Codec):
    def encode(self, input: str, errors: str = "strict") -> tuple[bytes, int]: ...
    def decode(self, input: bytes, errors: str = "strict") -> tuple[str, int]: ...

class IncrementalEncoder(codecs.IncrementalEncoder):
    def encode(self, input: str, final: bool = False) -> bytes: ...

class IncrementalDecoder(codecs.IncrementalDecoder):
    def decode(self, input: ReadableBuffer, final: bool = False) -> str: ...

class StreamWriter(Codec, codecs.StreamWriter): ...
class StreamReader(Codec, codecs.StreamReader): ...

def getregentry() -> codecs.CodecInfo: ...

decoding_map: dict[int, int | None]
decoding_table: str
encoding_map: dict[int, int]
