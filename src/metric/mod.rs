pub mod simd;
pub mod utils;

pub trait Metric : Clone {
    fn get_type() -> MetricType;
    fn compute(a: &[f32], b: &[f32]) -> f32;
}

#[derive(Clone, Copy, Debug)]
pub enum MetricType {
    Euclidean,
    DotProduct,
}

#[derive(Clone)]
pub struct EuclideanMetric;
#[derive(Clone)]
pub struct DotProductMetric;

impl Metric for EuclideanMetric {
    fn get_type() -> MetricType {
        MetricType::Euclidean
    }

    fn compute(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
    }
}

impl Metric for DotProductMetric {
    fn get_type() -> MetricType {
        MetricType::DotProduct
    }

    fn compute(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
    }
}