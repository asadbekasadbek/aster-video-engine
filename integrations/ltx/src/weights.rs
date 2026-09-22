use candle_core::{DType, Device, Tensor};
use candle_video::models::ltx_video::weight_format::KeyRemapper;
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub fn verify(path: &Path, size: u64, expected: &str) -> Result<()> {
    let mut file = File::open(path)?;
    if !file.metadata()?.is_file() || file.metadata()?.len() != size {
        return Err(format!("wrong file type or size: {}", path.display()).into());
    }
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    if format!("{:x}", hash.finalize()) != expected {
        return Err(format!("SHA-256 mismatch: {}", path.display()).into());
    }
    Ok(())
}

/// Reads ONLY selected component payloads, one tensor at a time. Retained GPU
/// weight bytes and a 256 MiB single-tensor bound are checked before allocation.
/// Backend activation/workspace overhead is additional, not enforced by this cap.
pub fn load(
    path: &Path,
    vae: bool,
    dtype: DType,
    device: &Device,
    limit: usize,
) -> Result<HashMap<String, Tensor>> {
    let mut file = File::open(path)?;
    let inspection = aster_core::model::inspect(&mut file)?;
    load_from(&mut file, &inspection, vae, dtype, device, limit)
}

fn load_from(
    reader: &mut (impl Read + Seek),
    inspection: &aster_core::model::Inspection,
    vae: bool,
    dtype: DType,
    device: &Device,
    limit: usize,
) -> Result<HashMap<String, Tensor>> {
    let remapper = KeyRemapper::new();
    let mut total = 0_usize;
    let mut selected = Vec::new();
    for key in inspection.metadata.offset_keys() {
        if if vae {
            key.starts_with("vae.")
        } else {
            key.starts_with("model.diffusion_model.")
        } {
            let info = inspection
                .metadata
                .info(&key)
                .ok_or("missing tensor metadata")?;
            let input_dtype = match info.dtype {
                safetensors::Dtype::F32 => DType::F32,
                safetensors::Dtype::F16 => DType::F16,
                safetensors::Dtype::BF16 => DType::BF16,
                _ => return Err("unsupported weight dtype; no implicit FP8 conversion".into()),
            };
            let bytes = info.data_offsets.1 - info.data_offsets.0;
            if bytes > 256 * 1024 * 1024 {
                return Err("single weight tensor exceeds 256 MiB staging budget".into());
            }
            let retained = (bytes / input_dtype.size_in_bytes())
                .checked_mul(dtype.size_in_bytes())
                .ok_or("weight size overflow")?;
            total = total
                .checked_add(retained)
                .ok_or("weight budget overflow")?;
            if total > limit {
                return Err("component exceeds configured weight budget".into());
            }
            selected.push((key, input_dtype));
        }
    }
    if selected.is_empty() {
        return Err("selected model component is absent".into());
    }
    let mut tensors = HashMap::new();
    let mut residency = aster_core::residency::Manager::new([0, 512 * 1024 * 1024, limit]);
    for (id, (key, input_dtype)) in selected.into_iter().enumerate() {
        let info = inspection
            .metadata
            .info(&key)
            .ok_or("missing tensor metadata")?;
        let mut raw = vec![0; info.data_offsets.1 - info.data_offsets.0];
        reader.seek(SeekFrom::Start(
            8 + inspection.header_bytes + info.data_offsets.0 as u64,
        ))?;
        reader.read_exact(&mut raw)?;
        // Convert on CPU, then transfer. No full-file mmap or whole-model GPU map.
        let cpu = Tensor::from_raw_buffer(&raw, input_dtype, &info.shape, &Device::Cpu)?
            .to_dtype(dtype)?;
        drop(raw);
        let tensor = if device.is_cuda() {
            use aster_core::residency::{Resource, Tier};
            residency.register(Resource {
                id: id as u32,
                bytes: cpu.elem_count() * dtype.size_in_bytes(),
                tier: Tier::Ram,
            })?;
            let mut source = Some(cpu);
            residency.transfer_value(id as u32, Tier::Vram, 0, |_, _| -> Result<Tensor> {
                let destination = source
                    .as_ref()
                    .ok_or("missing CPU tensor")?
                    .to_device(device)?;
                device.synchronize()?;
                drop(source.take());
                Ok(destination)
            })?
        } else {
            cpu.to_device(device)?
        };
        let remapped = remapper.remap_key(&key);
        let clean = remapped
            .strip_prefix("vae.")
            .or_else(|| remapped.strip_prefix("model.diffusion_model."))
            .or_else(|| remapped.strip_prefix("transformer."))
            .unwrap_or(&remapped);
        if tensors.insert(clean.to_owned(), tensor).is_some() {
            return Err("duplicate remapped tensor key".into());
        }
    }
    println!(
        "stage={} weight_payload_bytes={total}",
        if vae { "vae_load" } else { "dit_load" }
    );
    if device.is_cuda() {
        println!(
            "residency_scope=component_weight_transfer reserved_peak_bytes={:?} resident_bytes={:?}",
            residency.reserved_peak(),
            residency.used()
        );
    }
    Ok(tensors)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn loads_only_selected_component_and_enforces_budget() {
        // VAE time embeddings must not match the transformer's broad upstream heuristic.
        let header = r#"{"model.diffusion_model.w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"vae.time_embed.w":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header.as_bytes());
        bytes.extend(3_f32.to_le_bytes());
        bytes.extend(7_f32.to_le_bytes());
        let mut reader = std::io::Cursor::new(bytes);
        let metadata = aster_core::model::inspect(&mut reader).unwrap();
        let tensors =
            load_from(&mut reader, &metadata, false, DType::F32, &Device::Cpu, 4).unwrap();
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors["w"].to_vec1::<f32>().unwrap(), [3.]);
        let tensors = load_from(&mut reader, &metadata, true, DType::F32, &Device::Cpu, 4).unwrap();
        assert_eq!(tensors["time_embed.w"].to_vec1::<f32>().unwrap(), [7.]);
        assert!(load_from(&mut reader, &metadata, false, DType::F32, &Device::Cpu, 3).is_err());
    }
}
