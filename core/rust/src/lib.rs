#[derive(Debug, Clone, Copy)]
pub struct Neuron {
    pub state: f32,
    pub threshold: f32,
    pub leak: f32,
}

impl Neuron {
    pub fn new(threshold: f32, leak: f32) -> Self {
        Self {
            state: 0.0,
            threshold,
            leak,
        }
    }

    pub fn stimulate(&mut self, input: f32) {
        self.state += input;
    }

    pub fn step(&mut self) -> bool {
        self.state *= self.leak;

        if self.state >= self.threshold {
            self.state = 0.0;
            true
        } else {
            false
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Connection {
    pub from: usize,
    pub to: usize,
    pub weight: f32,
}

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;

        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;

        self.state = x;
        x
    }

    fn next_usize(&mut self, max: usize) -> usize {
        (self.next_u64() as usize) % max
    }
}

pub struct Brain {
    pub neurons: Vec<Neuron>,
    pub connections: Vec<Connection>,
}

impl Brain {
    pub fn new(neuron_count: usize) -> Self {
        let neurons = (0..neuron_count)
            .map(|_| Neuron::new(1.0, 1.0))
            .collect();

        Self {
            neurons,
            connections: Vec::new(),
        }
    }

    pub fn generate(
        neuron_count: usize,
        seed: u64,
        connections_per_neuron: usize,
    ) -> Self {
        let mut brain = Self::new(neuron_count);
        let mut rng = SimpleRng::new(seed);

        if neuron_count == 0 {
            return brain;
        }

        for from in 0..neuron_count {
            for _ in 0..connections_per_neuron {
                let mut to = rng.next_usize(neuron_count);

                while to == from && neuron_count > 1 {
                    to = rng.next_usize(neuron_count);
                }

                brain.connect(from, to, 0.5);
            }
        }

        brain
    }

    pub fn connect(&mut self, from: usize, to: usize, weight: f32) {
        self.connections.push(Connection { from, to, weight });
    }

    pub fn stimulate(&mut self, neuron: usize, amount: f32) {
        self.neurons[neuron].stimulate(amount);
    }

    pub fn step(&mut self) {
        let mut fired = Vec::new();

        for (index, neuron) in self.neurons.iter_mut().enumerate() {
            if neuron.step() {
                fired.push(index);
            }
        }

        for connection in &self.connections {
            if fired.contains(&connection.from) {
                self.neurons[connection.to]
                    .stimulate(connection.weight);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neuron_can_fire() {
        let mut neuron = Neuron::new(1.0, 1.0);
        neuron.stimulate(1.0);

        assert!(neuron.step());
        assert_eq!(neuron.state, 0.0);
    }

    #[test]
    fn neuron_does_not_fire_below_threshold() {
        let mut neuron = Neuron::new(1.0, 1.0);
        neuron.stimulate(0.5);

        assert!(!neuron.step());
        assert_eq!(neuron.state, 0.5);
    }

    #[test]
    fn brain_can_create_neurons() {
        let brain = Brain::new(100);

        assert_eq!(brain.neurons.len(), 100);
    }

    #[test]
    fn brain_can_create_connection() {
        let mut brain = Brain::new(100);
        brain.connect(0, 1, 0.5);

        assert_eq!(brain.connections.len(), 1);
        assert_eq!(brain.connections[0].from, 0);
        assert_eq!(brain.connections[0].to, 1);
    }

    #[test]
    fn signal_travels_between_neurons() {
        let mut brain = Brain::new(2);

        brain.connect(0, 1, 1.0);
        brain.stimulate(0, 1.0);
        brain.step();

        assert_eq!(brain.neurons[0].state, 0.0);
        assert_eq!(brain.neurons[1].state, 1.0);
    }

    #[test]
    fn brain_can_generate_connections() {
        let brain = Brain::generate(100, 123, 3);

        assert_eq!(brain.neurons.len(), 100);
        assert_eq!(brain.connections.len(), 300);
    }

    #[test]
    fn generation_is_deterministic() {
        let brain_a = Brain::generate(100, 123, 3);
        let brain_b = Brain::generate(100, 123, 3);

        for (a, b) in brain_a.connections.iter().zip(brain_b.connections.iter()) {
            assert_eq!(a.from, b.from);
            assert_eq!(a.to, b.to);
            assert_eq!(a.weight, b.weight);
        }
    }
}