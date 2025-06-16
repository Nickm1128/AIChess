import random 
import numpy as np

# --- Neuron and Synapse Classes ---
class Neuron:
    def __init__(self, name):
        self.name = name
        self.state = 0.0
        self.threshold_pos = random.uniform(0.05, 0.5)
        self.threshold_neg = random.uniform(-0.5, -0.05)
        self.fired = False
        

    def reset(self):
        self.state = 0.0
        self.fired = False

    def receive(self, value):
        self.state += value
        self.state = max(-1.0, min(1.0, self.state))

    def update(self):
        if self.state >= self.threshold_pos:
            self.fired = True
            return 'excite'
        elif self.state <= self.threshold_neg:
            self.fired = True
            return 'inhibit'
        else:
            self.fired = False
            return None


class Synapse:
    def __init__(self, pre, post, weight):
        self.pre = pre
        self.post = post
        self.weight = weight
        self.learning_rate = 0.1

    def propagate(self):
        if self.pre.fired:
            signal = 1.0 if self.pre.state > 0 else -1.0
            self.post.receive(signal * self.weight)

    def update_weight(self, reward):
        # Now 'reward' is a float, not just boolean. Adjust delta accordingly.
        if self.pre.fired and self.post.fired:
            # delta = self.learning_rate if reward else -self.learning_rate # Original binary logic
            delta = self.learning_rate * reward # Scale learning rate by reward magnitude
            self.weight = max(min(self.weight + delta, 2.0), -2.0)


class Agent:
    def __init__(self, name, neuron_count):
        self.name = name
        self.neurons = [Neuron(f"{name}_n{i}") for i in range(neuron_count)]
        self.synapses = []
        self.energy = 100.0
        self.position = (0, 0)

        self.initialize_synapses()

        if neuron_count < 3:
            raise ValueError("Agent must have at least 2 neurons for output actions.")
        self.output_neurons = self.neurons[-4:]

    def initialize_synapses(self):
        self.synapses = []
        for _ in range(len(self.neurons) * 10):
            pre, post = random.sample(self.neurons, 2) if len(self.neurons) >= 2 else (self.neurons[0], self.neurons[0])
            weight = random.uniform(-1, 1)
            self.synapses.append(Synapse(pre, post, weight))

    def reset(self):
        for n in self.neurons:
            n.reset()

    def step(self, think=1):
        for round in range(think):
            for syn in self.synapses:
                syn.propagate()
            for n in self.neurons:
                n.update()

    def learn(self, reward_value): # Renamed 'won' to 'reward_value' for clarity
        for syn in self.synapses:
            syn.update_weight(reward_value) # Pass the numerical reward


    def receive_inputs(self, inputs):
        for i, value in enumerate(inputs):
            if i < len(self.neurons):
                self.neurons[i].receive(value)

    def decide_action(self):
        output_mean = np.mean([n.state for n in self.output_neurons])

        if output_mean < -0.5:
            return 'jump_left'
        elif -0.5 <= output_mean < 0:
            return 'left'
        elif 0 < output_mean <= 0.5:
            return 'right'
        elif output_mean > 0.5:
            return 'jump_right'
        else:
            return 'wait'