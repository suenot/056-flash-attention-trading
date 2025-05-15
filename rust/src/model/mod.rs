//! Model module for Flash Attention trading.
//!
//! Contains transformer architecture and trading model implementations.

mod trading;
mod transformer;

pub use trading::{FlashAttentionTrader, OutputType, TraderConfig};
pub use transformer::{
    FeedForward, LayerNorm, MultiHeadAttention, PositionalEncoding, TransformerBlock,
    TransformerConfig,
};
