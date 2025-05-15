//! Trading signal generation.
//!
//! Converts model predictions to trading signals.

use ndarray::Array1;

/// Trading signal type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradingSignal {
    Buy,
    Sell,
    Hold,
}

impl TradingSignal {
    /// Convert to position multiplier
    pub fn to_position(&self) -> f64 {
        match self {
            TradingSignal::Buy => 1.0,
            TradingSignal::Sell => -1.0,
            TradingSignal::Hold => 0.0,
        }
    }
}

/// Signal generator configuration
#[derive(Debug, Clone)]
pub struct SignalGenerator {
    /// Threshold for buy signal (positive prediction)
    pub buy_threshold: f32,
    /// Threshold for sell signal (negative prediction)
    pub sell_threshold: f32,
    /// Whether to allow short positions
    pub allow_short: bool,
    /// Maximum position size (as fraction of capital)
    pub max_position: f64,
}

impl Default for SignalGenerator {
    fn default() -> Self {
        Self {
            buy_threshold: 0.001,  // 0.1% expected return
            sell_threshold: -0.001,
            allow_short: true,
            max_position: 1.0,
        }
    }
}

impl SignalGenerator {
    /// Generate signal from a single prediction
    pub fn generate(&self, prediction: f32) -> TradingSignal {
        if prediction > self.buy_threshold {
            TradingSignal::Buy
        } else if prediction < self.sell_threshold && self.allow_short {
            TradingSignal::Sell
        } else {
            TradingSignal::Hold
        }
    }

    /// Generate signals from prediction array
    pub fn generate_batch(&self, predictions: &Array1<f32>) -> Vec<TradingSignal> {
        predictions.iter().map(|&p| self.generate(p)).collect()
    }

    /// Calculate position size based on prediction confidence
    pub fn position_size(&self, prediction: f32) -> f64 {
        let base_size = if prediction > self.buy_threshold {
            (prediction - self.buy_threshold) as f64 / self.buy_threshold as f64
        } else if prediction < self.sell_threshold {
            (self.sell_threshold - prediction) as f64 / self.sell_threshold.abs() as f64
        } else {
            0.0
        };

        // Clip to max position
        base_size.min(self.max_position).max(-self.max_position)
    }
}

/// Convenience function to generate signals
pub fn generate_signals(predictions: &Array1<f32>, buy_threshold: f32, sell_threshold: f32) -> Vec<TradingSignal> {
    let generator = SignalGenerator {
        buy_threshold,
        sell_threshold,
        ..Default::default()
    };
    generator.generate_batch(predictions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_generation() {
        let generator = SignalGenerator::default();

        assert_eq!(generator.generate(0.01), TradingSignal::Buy);
        assert_eq!(generator.generate(-0.01), TradingSignal::Sell);
        assert_eq!(generator.generate(0.0), TradingSignal::Hold);
    }

    #[test]
    fn test_batch_signals() {
        let generator = SignalGenerator {
            buy_threshold: 0.005,
            sell_threshold: -0.005,
            ..Default::default()
        };

        let predictions = Array1::from_vec(vec![0.01, -0.01, 0.0, 0.02, -0.02]);
        let signals = generator.generate_batch(&predictions);

        assert_eq!(signals[0], TradingSignal::Buy);
        assert_eq!(signals[1], TradingSignal::Sell);
        assert_eq!(signals[2], TradingSignal::Hold);
    }

    #[test]
    fn test_no_short() {
        let generator = SignalGenerator {
            allow_short: false,
            ..Default::default()
        };

        assert_eq!(generator.generate(-0.01), TradingSignal::Hold);
    }
}
