import torch
from model_wrapper import ModelWrapper
from dataset_handler import DatasetHandler
from typing import List, Dict
import random 

class SteeringVector:
    

    def __init__(self, model_handler: ModelWrapper,dataset_handler: DatasetHandler, layer: int, method: str):
        self.activation_cache = {}
        self.dataset_handler = dataset_handler
        self.model_handler = model_handler
        self.layer = layer
        self.method = method
    
    def train(self, activations = None):
        '''can train with activations given in the required format'''
        if self.method == "caa":
            if activations is None:
                self.pos_data = self.dataset_handler.format_dataset(assist_key = "answer_matching_behavior")
                self.neg_data = self.dataset_handler.format_dataset(assist_key = "answer_not_matching_behavior")
                self.vec = self.train_caa(self.pos_data, self.neg_data, activations = False)
            else: # activations is {pos_acts: pos_acts tensor, neg_acts: neg_acts tensor}
                self.vec = self.train_caa(activations["pos_acts"], activations["neg_acts"], activations = True)
        
    
    def train_caa(self, pos_data, neg_data, activations = False, avg = True):
        '''train steering vector using caa
        if activations are False, will compute activations; expect pos_data & neg_data to be DatasetHandler objects
        if activations are True, then pos_data & neg_data are activations, not prompts'''
        if not activations:
            pos_acts = get_mean_activations(self.model_handler, pos_data, [self.layer], avg = avg)
            neg_acts = get_mean_activations(self.model_handler, neg_data, [self.layer], avg = avg)
            return pos_acts - neg_acts
        else:
            if avg: # average over all the datapoints 
                return torch.mean(pos_data - neg_data, dim = 0)
            else:
                return pos_data - neg_data
            
    def gen_steering_hook(self, multiplier):
        """
        A hook function that adds a steering vector to the module's output.
        This version is robust for use with model.generate().
        """
        id = random.randint(0, 9999)
        print(f"Hook generatd: ID {id}")
        def steering_hook(model, input, output):
            # print(f"Hook ID {id} called with multiplier {multiplier}")
            hidden_state = output[0].detach().clone()
            modified_activation = hidden_state[:, -1, :] + (self.vec.to(hidden_state.device) * multiplier)
            hidden_state[:, -1, :] = modified_activation
            assert hidden_state.shape == output[0].shape

            return (hidden_state,) + output[1:]
            
        return steering_hook
    
    def activate_stering(self, multiplier, clear_other_hooks = True):
        if clear_other_hooks:
            self.model_handler.clear_hooks()
        steering_hook_fn = self.gen_steering_hook(multiplier)
        self.steering_hook = self.model_handler.add_hook(steering_hook_fn, self.layer)

    def deactivate_steering(self):
        self.model_handler.remove_hook(self.steering_hook)
        self.steering_hook = None

        


activation_cache = {}
def get_activation_hook(name):
    def hook(model, input, output):
        activation_cache[name] = output[0].detach()
    return hook

def get_mean_activations(model_handler: ModelWrapper, dataset_handler: DatasetHandler, layers: List[int], token_pos = -1, avg = True):
    # add hooks for getting activations
    hooks = {}
    for layer in layers:
        hook_handle = model_handler.add_hook(get_activation_hook(f"{layer}_activation"), layer)
        hooks[layer] = hook_handle
    all_activations = {layer: [] for layer in layers} # store activations in dict {layer_idx:activations}
    
    for prompt in dataset_handler.train_loader:
        with torch.no_grad():
            model_handler.model(prompt["input_ids"], prompt["attention_mask"]) # double check this is correct
        
        for layer in layers:
            if token_pos is None: # if no token_pos given, average across all tokens
                mean_acts = torch.mean(activation_cache[f"{layer}_activation"], dim = 1)
                all_activations[layer].append(mean_acts)
            else:
                all_activations[layer].append(activation_cache[f"{layer}_activation"][:, token_pos, :])
    
    for layer, acts in all_activations.items():
        model_handler.remove_hook(hooks[layer]) # remove hook for getting activations from that layer
        if avg:
            all_activations[layer] = torch.mean(torch.cat(all_activations[layer], dim = 0), dim = 0)
        else:
            all_activations[layer] = torch.cat(all_activations[layer], dim = 0)
    
    return all_activations
            

def eval_steering(steering_vec: SteeringVector, test_dataset, method = ""):
    pass





