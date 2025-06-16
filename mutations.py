import random
import copy
from Agent import Synapse, Agent, Neuron

def mutate_agent(original_agent, mutation_rate, mutation_strength):
    """
    Creates a mutated copy of an agent, allowing for structural changes (adding neurons),
    modifying its parameters, and potentially pruning.
    """
    mutated_agent = copy.deepcopy(original_agent)

    # --- Structural Mutation: Add Neurons ---
    ADD_NEURON_CHANCE = 0.05 # 5% chance to add new neurons
    MAX_NEURONS_TO_ADD = 2 # Add 1 or 2 new neurons

    if random.random() < ADD_NEURON_CHANCE:
        num_neurons_to_add = random.randint(1, MAX_NEURONS_TO_ADD)
        current_neuron_count = len(mutated_agent.neurons)
        for i in range(num_neurons_to_add):
            new_neuron_name = f"{mutated_agent.name}_n_new{current_neuron_count + i}"
            new_neuron = Neuron(new_neuron_name)
            mutated_agent.neurons.append(new_neuron)
            
            # Add a few random synapses involving the newly added neuron
            # Connect the new neuron randomly within the existing/growing network
            num_new_synapses_for_neuron = random.randint(1, 5) # Add 1 to 5 synapses per new neuron
            for _ in range(num_new_synapses_for_neuron):
                if len(mutated_agent.neurons) >= 2: # Ensure enough neurons for sampling
                    pre, post = random.sample(mutated_agent.neurons, 2)
                    weight = random.uniform(-1, 1)
                    mutated_agent.synapses.append(Synapse(pre, post, weight))
    
    # --- Pruning Logic (Existing) ---
    PRUNING_CHANCE = 0.1 
    PRUNING_NEURON_MAX = 1
    PRUNING_SYNAPSE_MAX_PERCENT = 0.05

    MIN_NEURONS = 4 

    # 1. Prune Neurons
    if random.random() < PRUNING_CHANCE:
        neurons_to_prune_count = random.randint(0, PRUNING_NEURON_MAX)
        actual_neurons_to_prune = min(neurons_to_prune_count, len(mutated_agent.neurons) - MIN_NEURONS)
        
        if actual_neurons_to_prune > 0:
            non_output_neurons = [n for n in mutated_agent.neurons if n not in mutated_agent.output_neurons]
            
            if len(non_output_neurons) > 0:
                neurons_to_remove = random.sample(non_output_neurons, min(actual_neurons_to_prune, len(non_output_neurons)))
                
                for neuron_to_remove in neurons_to_remove:
                    mutated_agent.neurons.remove(neuron_to_remove)
                    mutated_agent.synapses = [
                        syn for syn in mutated_agent.synapses 
                        if syn.pre != neuron_to_remove and syn.post != neuron_to_remove
                    ]
                # print(f"Pruned {len(neurons_to_remove)} neurons from {original_agent.name}. New neuron count: {len(mutated_agent.neurons)}")


    # 2. Prune Synapses (independent of neuron pruning)
    if random.random() < PRUNING_CHANCE:
        num_synapses_to_prune = int(len(mutated_agent.synapses) * PRUNING_SYNAPSE_MAX_PERCENT)
        
        if num_synapses_to_prune > 0:
            synapses_to_remove = random.sample(mutated_agent.synapses, min(num_synapses_to_prune, len(mutated_agent.synapses)))
            for syn in synapses_to_remove:
                mutated_agent.synapses.remove(syn)
            # print(f"Pruned {len(synapses_to_remove)} synapses from {original_agent.name}. New synapse count: {len(mutated_agent.synapses)}")


    # --- Apply parameter mutations (weights and thresholds) ---
    for syn in mutated_agent.synapses:
        if random.random() < mutation_rate:
            syn.weight += random.uniform(-mutation_strength, mutation_strength)
            syn.weight = max(min(syn.weight, 2.0), -2.0)

    for neuron in mutated_agent.neurons:
        if random.random() < mutation_rate:
            neuron.threshold_pos += random.uniform(-0.1, 0.1)
            neuron.threshold_neg += random.uniform(-0.1, 0.1)
            neuron.threshold_pos = max(0.01, min(1.0, neuron.threshold_pos))
            neuron.threshold_neg = min(-0.01, max(-1.0, neuron.threshold_neg))

    mutated_agent.name = 'Child' + f"_mut{random.randint(0, 9999)}_NEURONCOUNT{len(mutated_agent.neurons)}"

    return mutated_agent

def crossover_agents(parent1, parent2, neuron_count):
    """
    Creates a child agent by combining synapses from two parent agents.
    Assumes parents have the same number of neurons and similar synapse structure.
    """
    child_agent = Agent("Crossover_Child", neuron_count)
    child_agent.synapses = [] # Clear initial random synapses for child

    # Simple uniform crossover: for each synapse, randomly pick from parent1 or parent2
    num_synapses = max(len(parent1.synapses), len(parent2.synapses))

    for i in range(num_synapses):
        chosen_synapse_parent = None
        if random.random() < 0.5: # 50% chance to inherit from parent1 or parent2
            if i < len(parent1.synapses):
                chosen_synapse_parent = parent1.synapses[i]
            elif i < len(parent2.synapses): # Fallback if parent1 has fewer synapses
                chosen_synapse_parent = parent2.synapses[i]
        else:
            if i < len(parent2.synapses):
                chosen_synapse_parent = parent2.synapses[i]
            elif i < len(parent1.synapses): # Fallback if parent2 has fewer synapses
                chosen_synapse_parent = parent1.synapses[i]

        if chosen_synapse_parent:
            pre_neuron_index = 0
            post_neuron_index = 0

            if chosen_synapse_parent in parent1.synapses:
                pre_neuron_index = parent1.neurons.index(chosen_synapse_parent.pre)
                post_neuron_index = parent1.neurons.index(chosen_synapse_parent.post)
            elif chosen_synapse_parent in parent2.synapses:
                pre_neuron_index = parent2.neurons.index(chosen_synapse_parent.pre)
                post_neuron_index = parent2.neurons.index(chosen_synapse_parent.post)
            else:
                print("Warning: Chosen synapse parent not found in either parent's synapse list during crossover.")
                continue # Skip this synapse

            pre_neuron_child = child_agent.neurons[pre_neuron_index]
            post_neuron_child = child_agent.neurons[post_neuron_index]

            child_agent.synapses.append(Synapse(pre_neuron_child, post_neuron_child, chosen_synapse_parent.weight))

    while len(child_agent.synapses) < len(parent1.synapses): 
        pre, post = random.sample(child_agent.neurons, 2)
        weight = random.uniform(-1, 1)
        child_agent.synapses.append(Synapse(pre, post, weight))

    return child_agent