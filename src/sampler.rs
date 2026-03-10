//! 采样器构建模块
//!
//! 采样器顺序遵循 llama.cpp 推荐：过滤采样器（min_p/top_k/top_p）在温度缩放之前，
//! 确保阈值计算基于原始概率分布而非被温度扭曲的分布。

use crate::config::{LlmConfig, SamplingMode};
use llama_cpp_2::sampling::LlamaSampler;

/// 根据配置构建采样器
pub(crate) fn build_sampler(config: &LlmConfig) -> LlamaSampler {
    match &config.sampling_mode {
        SamplingMode::Deterministic => LlamaSampler::chain_simple([LlamaSampler::greedy()]),
        SamplingMode::Creative { temperature, min_p } => LlamaSampler::chain_simple([
            LlamaSampler::min_p(*min_p, 1),
            LlamaSampler::temp(*temperature),
            LlamaSampler::dist(config.seed),
        ]),
        SamplingMode::Agent { temperature, min_p } => LlamaSampler::chain_simple([
            LlamaSampler::min_p(*min_p, 1),
            LlamaSampler::temp(*temperature),
            LlamaSampler::dist(config.seed),
        ]),
        SamplingMode::Custom {
            temperature,
            min_p,
            top_p,
            top_k,
            seed,
        } => {
            let mut samplers = Vec::new();
            // 过滤采样器先于温度缩放：top_k → top_p → min_p → temp → dist
            if let Some(k) = top_k {
                samplers.push(LlamaSampler::top_k(*k));
            }
            if let Some(p) = top_p {
                samplers.push(LlamaSampler::top_p(*p, 1));
            }
            if let Some(p) = min_p {
                samplers.push(LlamaSampler::min_p(*p, 1));
            }
            samplers.push(LlamaSampler::temp(*temperature));
            samplers.push(LlamaSampler::dist(*seed));
            LlamaSampler::chain(samplers, false)
        }
    }
}

/// 构建带 grammar 约束的采样器
///
/// grammar 约束场景使用 greedy 保证输出严格符合语法
pub(crate) fn build_grammar_sampler(grammar_sampler: LlamaSampler) -> LlamaSampler {
    LlamaSampler::chain_simple([grammar_sampler, LlamaSampler::greedy()])
}
