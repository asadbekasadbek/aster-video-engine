pub mod controlled;
pub mod latent;
mod portrait_attention;
pub mod scene;
pub mod transformer;

#[cfg(test)]
mod tests {
    use super::transformer::LtxVideoTransformer3DModel as Adapted;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};
    use candle_video::models::ltx_video::ltx_transformer::{
        LtxVideoTransformer3DModel as Original, LtxVideoTransformer3DModelConfig,
    };

    #[test]
    fn token_time_preserves_scalar_path_and_changes_conditioning() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let config = LtxVideoTransformer3DModelConfig {
            in_channels: 4,
            out_channels: 4,
            caption_channels: 12,
            num_layers: 1,
            num_attention_heads: 1,
            attention_head_dim: 12,
            cross_attention_dim: 12,
            ..Default::default()
        };
        let variables = VarMap::new();
        let vb = VarBuilder::from_varmap(&variables, DType::F32, &device);
        let original = Original::new(&config, vb.clone())?;
        let adapted = Adapted::new(&config, vb)?;
        let input = Tensor::ones((1, 2, 4), DType::F32, &device)?;
        let text = Tensor::ones((1, 3, 12), DType::F32, &device)?;
        let scalar = Tensor::new(&[500_f32], &device)?;
        let repeated = Tensor::new(&[[500_f32, 500.]], &device)?;
        let mixed = Tensor::new(&[[0_f32, 500.]], &device)?;
        let baseline = original.forward(&input, &text, &scalar, None, 1, 1, 2, None, None, None)?;
        let a = adapted.forward(&input, &text, &scalar, None, 1, 1, 2, None, None, None)?;
        let b = adapted.forward(&input, &text, &repeated, None, 1, 1, 2, None, None, None)?;
        let c = adapted.forward(&input, &text, &mixed, None, 1, 1, 2, None, None, None)?;
        assert!((&baseline - &a)?.abs()?.max_all()?.to_scalar::<f32>()? < 1e-5);
        assert!((&a - &b)?.abs()?.max_all()?.to_scalar::<f32>()? < 1e-5);
        assert!((&b - &c)?.abs()?.max_all()?.to_scalar::<f32>()? > 1e-7);
        let invalid = Tensor::zeros((2, 1), DType::F32, &device)?;
        assert!(
            adapted
                .forward(&input, &text, &invalid, None, 1, 1, 2, None, None, None)
                .is_err()
        );
        Ok(())
    }
}
