//! Header-only SafeTensors inspection. No graph execution, weights download or mmap.
use safetensors::tensor::Metadata;
use serde::de::{self, IgnoredAny, MapAccess, Visitor};
use std::collections::HashSet;
use std::fmt;
use std::io::{self, Read, Seek, SeekFrom};

pub const MAX_HEADER_BYTES: u64 = 1024 * 1024;
pub const MAX_TENSORS: usize = 4096;

fn invalid(message: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

// Safetensors' internal HashMap accepts repeated top-level names; reject them
// before parsing Metadata so an ambiguous archive cannot hide an earlier entry.
struct UniqueKeys;
impl<'de> Visitor<'de> for UniqueKeys {
    type Value = ();
    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a bounded object with unique tensor names")
    }
    fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<(), M::Error> {
        let mut names = HashSet::new();
        while let Some(name) = map.next_key::<String>()? {
            if name.len() > 4096 || names.len() > MAX_TENSORS {
                return Err(de::Error::custom("header name/count limit"));
            }
            if !names.insert(name) {
                return Err(de::Error::custom("duplicate tensor name"));
            }
            map.next_value::<IgnoredAny>()?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct Inspection {
    pub header_bytes: u64,
    pub data_bytes: u64,
    pub metadata: Metadata,
}

/// The reader must describe an immutable snapshot. Reads only the length prefix
/// and <=1 MiB JSON header; uses file length to validate the tensor data extent.
/// Payload contents, hashes, architecture and supported operators are NOT verified.
pub fn inspect(reader: &mut (impl Read + Seek)) -> io::Result<Inspection> {
    let file_len = reader.seek(SeekFrom::End(0))?;
    reader.seek(SeekFrom::Start(0))?;
    let mut length = [0; 8];
    reader.read_exact(&mut length)?;
    let header_bytes = u64::from_le_bytes(length);
    if header_bytes == 0 || header_bytes > MAX_HEADER_BYTES {
        return Err(invalid("header length outside 1..=1048576 bytes"));
    }
    let data_start = 8 + header_bytes;
    if data_start > file_len {
        return Err(invalid("truncated header"));
    }
    let mut header = vec![0; header_bytes as usize];
    reader.read_exact(&mut header)?;
    if header.first() != Some(&b'{') {
        return Err(invalid("header must start with an object"));
    }
    let mut deserializer = serde_json::Deserializer::from_slice(&header);
    serde::Deserializer::deserialize_map(&mut deserializer, UniqueKeys).map_err(invalid)?;
    deserializer.end().map_err(invalid)?;
    // Metadata::deserialize calls Metadata::new, validating shapes, dtype sizes,
    // contiguous offsets and arithmetic overflow in safetensors 0.8.0.
    let metadata: Metadata = serde_json::from_slice(&header).map_err(invalid)?;
    let names = metadata.offset_keys();
    if names.len() > MAX_TENSORS {
        return Err(invalid("too many tensors"));
    }
    for name in &names {
        if metadata.info(name).is_some_and(|t| t.shape.len() > 32) {
            return Err(invalid("tensor rank exceeds 32"));
        }
    }
    let data_bytes = metadata.data_len() as u64;
    if data_start.checked_add(data_bytes) != Some(file_len) {
        return Err(invalid("tensor data extent does not match file length"));
    }
    Ok(Inspection {
        header_bytes,
        data_bytes,
        metadata,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn archive(json: &str, data: &[u8]) -> Vec<u8> {
        let mut result = (json.len() as u64).to_le_bytes().to_vec();
        result.extend(json.as_bytes());
        result.extend(data);
        result
    }

    #[test]
    fn inspects_without_reading_payload() {
        let json = r#"{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let mut reader = Cursor::new(archive(json, &[0; 8]));
        let result = inspect(&mut reader).unwrap();
        assert_eq!(result.data_bytes, 8);
        assert_eq!(reader.position(), 8 + json.len() as u64);
        assert_eq!(result.metadata.info("weight").unwrap().shape, [2]);
        // Independent upstream full-buffer parser agrees for this fixture.
        assert_eq!(
            safetensors::SafeTensors::deserialize(reader.get_ref())
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn invalid_archives_are_rejected() {
        for (json, data) in [
            (
                r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
                vec![0; 4],
            ),
            (
                r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[1,9]}}"#,
                vec![0; 9],
            ),
            (
                r#"{"a":{"dtype":"UNKNOWN","shape":[1],"data_offsets":[0,4]}}"#,
                vec![0; 4],
            ),
            (
                r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
                vec![0; 4],
            ),
            (
                r#"{"a":{"dtype":"F32","shape":[18446744073709551615,2],"data_offsets":[0,0]}}"#,
                vec![],
            ),
            ("{}", vec![0]),
            ("{bad}", vec![]),
        ] {
            assert!(
                inspect(&mut Cursor::new(archive(json, &data))).is_err(),
                "{json}"
            );
        }
        for bytes in [
            vec![],
            vec![0; 7],
            (MAX_HEADER_BYTES + 1).to_le_bytes().to_vec(),
            u64::MAX.to_le_bytes().to_vec(),
            64_u64.to_le_bytes().to_vec(),
        ] {
            assert!(inspect(&mut Cursor::new(bytes)).is_err());
        }
    }

    #[test]
    fn empty_and_scalar_tensors_are_valid() {
        for (json, data) in [
            (
                r#"{"empty":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}}"#,
                vec![],
            ),
            (
                r#"{"scalar":{"dtype":"U8","shape":[],"data_offsets":[0,1]}}"#,
                vec![42],
            ),
        ] {
            assert!(inspect(&mut Cursor::new(archive(json, &data))).is_ok());
        }
    }

    #[test]
    fn names_count_rank_and_overlap_limits() {
        let entries: Vec<_> = (0..MAX_TENSORS + 1)
            .map(|i| format!(r#""t{i}":{{"dtype":"U8","shape":[0],"data_offsets":[0,0]}}"#))
            .collect();
        let too_many = format!("{{{}}}", entries.join(","));
        let rank = format!(
            r#"{{"t":{{"dtype":"U8","shape":[{}],"data_offsets":[0,1]}}}}"#,
            vec!["1"; 33].join(",")
        );
        let long_name = format!(
            r#"{{"{}":{{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}}}"#,
            "x".repeat(4097)
        );
        let overlap = r#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"b":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#;
        for json in [&too_many, &rank, &long_name, overlap] {
            assert!(inspect(&mut Cursor::new(archive(json, &[0]))).is_err());
        }
    }
}
