#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConsistencyMode {
    Strong,
    Eventual,
    Byzantine,
}

pub fn classify_temperature(temperature: f32) -> ConsistencyMode {
    if temperature <= 0.0 {
        ConsistencyMode::Strong
    } else if temperature < 1.5 {
        ConsistencyMode::Eventual
    } else {
        ConsistencyMode::Byzantine
    }
}
