use std::fs;
use std::path::Path;

use thiserror::Error;

use crate::gguf::quant::GgufQuantType;

const GGUF_MAGIC: [u8; 4] = *b"GGUF";
const DEFAULT_ALIGNMENT: u32 = 32;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ParserBootstrap;

impl ParserBootstrap {
    pub const SUPPORTED_VERSIONS: [u32; 2] = [2, 3];

    pub fn supported_versions(self) -> &'static [u32] {
        &Self::SUPPORTED_VERSIONS
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufHeader {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_count: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufMetadataValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl TryFrom<u32> for GgufMetadataValueType {
    type Error = ParseError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Uint8),
            1 => Ok(Self::Int8),
            2 => Ok(Self::Uint16),
            3 => Ok(Self::Int16),
            4 => Ok(Self::Uint32),
            5 => Ok(Self::Int32),
            6 => Ok(Self::Float32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::Uint64),
            11 => Ok(Self::Int64),
            12 => Ok(Self::Float64),
            other => Err(ParseError::UnsupportedMetadataType(other)),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GgufMetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array {
        element_type: GgufMetadataValueType,
        values: Vec<GgufMetadataValue>,
    },
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

#[derive(Debug, Clone, PartialEq)]
pub struct GgufMetadataEntry {
    pub key: String,
    pub value: GgufMetadataValue,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufTensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub raw_type_id: u32,
    pub data_offset: u64,
}

impl GgufTensorInfo {
    pub fn element_count(&self) -> u64 {
        self.dimensions.iter().copied().product::<u64>()
    }

    pub fn quant_type(&self) -> Option<GgufQuantType> {
        GgufQuantType::from_raw_ggml_type(self.raw_type_id)
    }

    pub fn absolute_offset(&self, tensor_data_offset: usize) -> Option<u64> {
        u64::try_from(tensor_data_offset)
            .ok()?
            .checked_add(self.data_offset)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GgufFile {
    pub header: GgufHeader,
    pub metadata: Vec<GgufMetadataEntry>,
    pub tensors: Vec<GgufTensorInfo>,
    pub alignment: u32,
    pub tensor_data_offset: usize,
}

impl GgufFile {
    pub fn find_metadata_value(&self, key: &str) -> Option<&GgufMetadataValue> {
        self.metadata
            .iter()
            .find(|entry| entry.key == key)
            .map(|entry| &entry.value)
    }

    pub fn tokenizer_metadata(&self) -> Vec<&GgufMetadataEntry> {
        self.metadata
            .iter()
            .filter(|entry| entry.key.starts_with("tokenizer.ggml"))
            .collect()
    }
}

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("failed to read GGUF file: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid GGUF magic: expected {expected:?}, found {found:?}")]
    InvalidMagic { expected: [u8; 4], found: [u8; 4] },
    #[error("unsupported GGUF version {0}")]
    UnsupportedVersion(u32),
    #[error("unexpected EOF while reading {context}: needed {needed} bytes, had {remaining}")]
    UnexpectedEof {
        context: &'static str,
        needed: usize,
        remaining: usize,
    },
    #[error("invalid UTF-8 in GGUF string: {0}")]
    InvalidUtf8(#[from] std::string::FromUtf8Error),
    #[error("unsupported GGUF metadata type {0}")]
    UnsupportedMetadataType(u32),
    #[error("numeric conversion overflow while reading {0}")]
    CountOverflow(&'static str),
    #[error("invalid GGUF alignment 0")]
    InvalidAlignment,
}

pub fn parse_path(path: impl AsRef<Path>) -> Result<GgufFile, ParseError> {
    let bytes = fs::read(path)?;
    parse_bytes(&bytes)
}

pub fn parse_bytes(bytes: &[u8]) -> Result<GgufFile, ParseError> {
    let mut cursor = Cursor::new(bytes);

    let magic = cursor.read_array::<4>("magic")?;
    if magic != GGUF_MAGIC {
        return Err(ParseError::InvalidMagic {
            expected: GGUF_MAGIC,
            found: magic,
        });
    }

    let version = cursor.read_u32("version")?;
    if !ParserBootstrap::SUPPORTED_VERSIONS.contains(&version) {
        return Err(ParseError::UnsupportedVersion(version));
    }

    let tensor_count = cursor.read_u64("tensor count")?;
    let metadata_count = cursor.read_u64("metadata count")?;

    let metadata_len =
        usize::try_from(metadata_count).map_err(|_| ParseError::CountOverflow("metadata count"))?;
    let tensor_len =
        usize::try_from(tensor_count).map_err(|_| ParseError::CountOverflow("tensor count"))?;

    let mut metadata = Vec::with_capacity(metadata_len);
    for _ in 0..metadata_len {
        let key = cursor.read_string("metadata key")?;
        let value_type = GgufMetadataValueType::try_from(cursor.read_u32("metadata value type")?)?;
        let value = parse_metadata_value(&mut cursor, value_type)?;
        metadata.push(GgufMetadataEntry { key, value });
    }

    let mut tensors = Vec::with_capacity(tensor_len);
    for _ in 0..tensor_len {
        let name = cursor.read_string("tensor name")?;
        let n_dimensions = cursor.read_u32("tensor dimensions")?;
        let dim_len = usize::try_from(n_dimensions)
            .map_err(|_| ParseError::CountOverflow("tensor dimensions"))?;

        let mut dimensions = Vec::with_capacity(dim_len);
        for _ in 0..dim_len {
            dimensions.push(cursor.read_u64("tensor shape")?);
        }

        let raw_type_id = cursor.read_u32("tensor type")?;
        let data_offset = cursor.read_u64("tensor data offset")?;

        tensors.push(GgufTensorInfo {
            name,
            dimensions,
            raw_type_id,
            data_offset,
        });
    }

    let alignment = match metadata
        .iter()
        .find(|entry| entry.key == "general.alignment")
        .map(|entry| &entry.value)
    {
        Some(GgufMetadataValue::Uint32(value)) => *value,
        Some(GgufMetadataValue::Uint64(value)) => {
            u32::try_from(*value).map_err(|_| ParseError::CountOverflow("general.alignment"))?
        }
        Some(_) | None => DEFAULT_ALIGNMENT,
    };

    if alignment == 0 {
        return Err(ParseError::InvalidAlignment);
    }

    let tensor_data_offset = align_offset(cursor.position(), alignment as usize);

    Ok(GgufFile {
        header: GgufHeader {
            version,
            tensor_count,
            metadata_count,
        },
        metadata,
        tensors,
        alignment,
        tensor_data_offset,
    })
}

fn parse_metadata_value(
    cursor: &mut Cursor<'_>,
    value_type: GgufMetadataValueType,
) -> Result<GgufMetadataValue, ParseError> {
    match value_type {
        GgufMetadataValueType::Uint8 => Ok(GgufMetadataValue::Uint8(cursor.read_u8("uint8")?)),
        GgufMetadataValueType::Int8 => Ok(GgufMetadataValue::Int8(cursor.read_i8("int8")?)),
        GgufMetadataValueType::Uint16 => Ok(GgufMetadataValue::Uint16(cursor.read_u16("uint16")?)),
        GgufMetadataValueType::Int16 => Ok(GgufMetadataValue::Int16(cursor.read_i16("int16")?)),
        GgufMetadataValueType::Uint32 => Ok(GgufMetadataValue::Uint32(cursor.read_u32("uint32")?)),
        GgufMetadataValueType::Int32 => Ok(GgufMetadataValue::Int32(cursor.read_i32("int32")?)),
        GgufMetadataValueType::Float32 => {
            Ok(GgufMetadataValue::Float32(cursor.read_f32("float32")?))
        }
        GgufMetadataValueType::Bool => Ok(GgufMetadataValue::Bool(cursor.read_bool("bool")?)),
        GgufMetadataValueType::String => {
            Ok(GgufMetadataValue::String(cursor.read_string("string")?))
        }
        GgufMetadataValueType::Array => {
            let element_type =
                GgufMetadataValueType::try_from(cursor.read_u32("array element type")?)?;
            let len = usize::try_from(cursor.read_u64("array length")?)
                .map_err(|_| ParseError::CountOverflow("array length"))?;
            let mut values = Vec::with_capacity(len);
            for _ in 0..len {
                values.push(parse_metadata_value(cursor, element_type)?);
            }
            Ok(GgufMetadataValue::Array {
                element_type,
                values,
            })
        }
        GgufMetadataValueType::Uint64 => Ok(GgufMetadataValue::Uint64(cursor.read_u64("uint64")?)),
        GgufMetadataValueType::Int64 => Ok(GgufMetadataValue::Int64(cursor.read_i64("int64")?)),
        GgufMetadataValueType::Float64 => {
            Ok(GgufMetadataValue::Float64(cursor.read_f64("float64")?))
        }
    }
}

fn align_offset(offset: usize, alignment: usize) -> usize {
    let remainder = offset % alignment;
    if remainder == 0 {
        offset
    } else {
        offset + (alignment - remainder)
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn position(&self) -> usize {
        self.offset
    }

    fn read_exact(&mut self, len: usize, context: &'static str) -> Result<&'a [u8], ParseError> {
        let end = self.offset.saturating_add(len);
        if end > self.bytes.len() {
            return Err(ParseError::UnexpectedEof {
                context,
                needed: len,
                remaining: self.bytes.len().saturating_sub(self.offset),
            });
        }

        let slice = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(slice)
    }

    fn read_array<const N: usize>(&mut self, context: &'static str) -> Result<[u8; N], ParseError> {
        let mut out = [0_u8; N];
        out.copy_from_slice(self.read_exact(N, context)?);
        Ok(out)
    }

    fn read_u8(&mut self, context: &'static str) -> Result<u8, ParseError> {
        Ok(self.read_exact(1, context)?[0])
    }

    fn read_i8(&mut self, context: &'static str) -> Result<i8, ParseError> {
        Ok(i8::from_le_bytes([self.read_u8(context)?]))
    }

    fn read_u16(&mut self, context: &'static str) -> Result<u16, ParseError> {
        Ok(u16::from_le_bytes(self.read_array(context)?))
    }

    fn read_i16(&mut self, context: &'static str) -> Result<i16, ParseError> {
        Ok(i16::from_le_bytes(self.read_array(context)?))
    }

    fn read_u32(&mut self, context: &'static str) -> Result<u32, ParseError> {
        Ok(u32::from_le_bytes(self.read_array(context)?))
    }

    fn read_i32(&mut self, context: &'static str) -> Result<i32, ParseError> {
        Ok(i32::from_le_bytes(self.read_array(context)?))
    }

    fn read_u64(&mut self, context: &'static str) -> Result<u64, ParseError> {
        Ok(u64::from_le_bytes(self.read_array(context)?))
    }

    fn read_i64(&mut self, context: &'static str) -> Result<i64, ParseError> {
        Ok(i64::from_le_bytes(self.read_array(context)?))
    }

    fn read_f32(&mut self, context: &'static str) -> Result<f32, ParseError> {
        Ok(f32::from_le_bytes(self.read_array(context)?))
    }

    fn read_f64(&mut self, context: &'static str) -> Result<f64, ParseError> {
        Ok(f64::from_le_bytes(self.read_array(context)?))
    }

    fn read_bool(&mut self, context: &'static str) -> Result<bool, ParseError> {
        Ok(self.read_u8(context)? != 0)
    }

    fn read_string(&mut self, context: &'static str) -> Result<String, ParseError> {
        let len = usize::try_from(self.read_u64(context)?)
            .map_err(|_| ParseError::CountOverflow("string length"))?;
        let bytes = self.read_exact(len, context)?;
        Ok(String::from_utf8(bytes.to_vec())?)
    }
}
