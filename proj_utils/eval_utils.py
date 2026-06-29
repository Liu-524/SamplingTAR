from tqdm import tqdm
import numpy as np

def get_top_activation_and_images(activation_to_top_activating_sample, selected_neurons, top_activation=1000, min_image=30, max_image=100):
        # activation_to_top_activating_sample: dict (neuron idx -> list of (activation, image idx))
        # selected_neurons: list of neuron idx
        # top_activation: int, number of neurons to return
        # top_image: int, number of top images per neuron to consider
        # returns:
        #   ret: list of list of image idx, shape (num_neurons, top_image)
        #   neurons: list of neuron idx, shape (num_neurons,)
        #   all_images: set of image idx, all unique images across all neurons


        if type(activation_to_top_activating_sample) == dict:
            activation = list(activation_to_top_activating_sample.values())
        else:
            activation = activation_to_top_activating_sample

        def topk_mean(x, k=20):
            if len(x) == 0:
                return -10
            return np.sum([a[0] for a in x[:k]])
        indices = [i for i in range(len(activation))]

        indices = sorted(indices, key=lambda x: topk_mean(activation[x], min_image), reverse=True)

        indices = [x for x in indices if len(activation[x]) >= min_image and len(activation[x]) <= max_image]
        activations = [activation[x] for x in indices]
        
        ret = [ [x[1] for x in activations[i][:min_image]] for i in range(min(top_activation, len(activations)))]
        neurons = [selected_neurons[i] for i in indices[:len(ret)]]
        all_images = set()
        for i in range(len(ret)):
            for j in range(min_image):
                all_images.add(ret[i][j])

        return ret, neurons, all_images

def neuron_dedup(
    activation_to_top_activating_sample, 
    scores=None,
    top_n=20, 
    similarity_threshold=0.2,
    target_num_neurons=-1,
):
    """
    This version uses a callable class, which is a notebook-friendly way
    to use multiprocessing without needing a separate helper file.
    """
    # Step 1: Pre-calculate sets (no change)
    print("Step 1: Pre-calculating all neuron sets...")
    if scores is not None:
        all_neuron_sets = sorted(activation_to_top_activating_sample.items(), key=lambda x: scores[x[0]], reverse=True)
    else:
        all_neuron_sets = sorted(activation_to_top_activating_sample.items(), key=lambda x: sum( a[0] for a in x[1][:top_n]), reverse=True)
    all_neuron_indices = [x[0] for x in all_neuron_sets]
    all_neuron_sets = [set(a[1] for a in x[1][:top_n]) for x in all_neuron_sets]
   
    print("...done.")
    
    # Step 2: Hybrid selection loop
    selected_indices = []
    selected_image_sets = []
    
    print(f"Step 2: Selecting unique neurons...")
    required_overlap = int(top_n * similarity_threshold)
    print(f"Required overlap for duplication: {required_overlap} images.")

    for i, current_set in tqdm(enumerate(all_neuron_sets)):
        if not current_set:
            continue
        num_selected = len(selected_image_sets)
        
        if num_selected == 0:
            selected_indices.append(all_neuron_indices[i])
            selected_image_sets.append(current_set)
            continue
        if len(current_set) < top_n:
            continue
        
        
        is_duplicate = False

        is_duplicate = any(
            len(current_set.intersection(existing_set)) >= required_overlap
            for existing_set in selected_image_sets
        )


        if not is_duplicate:
            selected_indices.append(all_neuron_indices[i])
            selected_image_sets.append(current_set)
            if target_num_neurons > 0 and len(selected_indices) >= target_num_neurons:
                break

    print("...selection complete.")
    return selected_indices