from src.integrated_information_theory.config.integrated_information_theory_config import integrated_information_theory_config
from sklearn import decomposition
from abc import ABC, abstractmethod
from ..utils.enums_class import tpm_creation_type_enum, last_layer_computation_type_enum, granularity_enum, iit_threashold_type_enum
from src.integrated_information_theory.entity.iit_token_entity import iit_token_entity
from src.integrated_information_theory.entity.iit_entity import iit_entity
import numpy as np
import pyphi
import math
import torch

class integrated_information_theory(ABC): 

    def __init__(self, config: integrated_information_theory_config):
        self.seed = 42
        self.config = config
        self.config.validate()

    def calculate(self, iit_entity_list: list[iit_entity]) -> list[iit_entity]: 
        if iit_entity_list is None or len(iit_entity_list) == 0:
            return []
        
        for entity in iit_entity_list:
            try:
                self.refine_embedding(entity)
            except Exception as e:
                print(f"[Error] calculate[refine_embedding]: {e}")

        calcutable_list = list(filter(lambda x: x.has_completion_embedding_for_pca() , iit_entity_list))
        if calcutable_list is None or len(calcutable_list) == 0:
            return []

        if tpm_creation_type_enum.TRAJECTORY == self.get_config().tpm_creation_type: 
            return self.calculate_per_trajectory(calcutable_list)
        elif tpm_creation_type_enum.PROMPT == self.get_config().tpm_creation_type:
            return self.calculate_per_prompt(calcutable_list)
        elif tpm_creation_type_enum.BATCH == self.get_config().tpm_creation_type:
            return self.calculate_iit_reward_mean_std(calcutable_list)
        else:
            return []

    def calculate_prompt(self, iit_entity_list: list[iit_entity]) -> iit_entity: 
        if iit_entity_list is None or len(iit_entity_list) == 0:
            return None
        
        for entity in iit_entity_list:
            try:
                self.refine_embedding(entity)
            except Exception as e:
                print(f"[Error] calculate[refine_embedding]: {e}")

        calcutable_list = list(filter(lambda x: x.has_completion_embedding_for_pca , iit_entity_list))
        if calcutable_list is None or len(calcutable_list) == 0:
            return None

        return self.calculate_iit_reward_prompt(calcutable_list)

    def calculate_entity(self, entity: iit_entity) -> iit_entity: 
        if entity is None or not entity.is_calcutable():
            return entity
        
        try:
            self.refine_embedding(entity)
        except Exception as e:
            print(f"[Error] calculate[refine_embedding]: {e}")
            return entity

        single_iit_entity_list = []
        single_iit_entity_list.append(entity)
        single_iit_entity_list = self.calculate_iit_reward_mean_std(single_iit_entity_list)
        return single_iit_entity_list[0]

    def calculate_per_trajectory(self, iit_entity_list: list[iit_entity]) -> list[iit_entity]: 
        new_iit_entity_list = []
        for entity in iit_entity_list:
            single_iit_entity_list = []
            single_iit_entity_list.append(entity)
            single_iit_entity_list = self.calculate_iit_reward_mean_std(single_iit_entity_list)
            new_iit_entity_list.extend(single_iit_entity_list)

        return new_iit_entity_list

    def calculate_per_prompt(self, iit_entity_list: list[iit_entity]) -> list[iit_entity]:
        prompt_set = set(map(lambda x: x.prompt_ID, iit_entity_list))
        new_iit_entity_list = []
        for prompt_ID in prompt_set:
            prompt_iit_entity_list = list(filter(lambda x: x.prompt_ID == prompt_ID, iit_entity_list))
            if len(prompt_iit_entity_list) == 0:
                continue

            new_prompt_iit_entity_list = self.calculate_iit_reward_mean_std(prompt_iit_entity_list)
            new_iit_entity_list.extend(new_prompt_iit_entity_list)

        return new_iit_entity_list

    def calculate_iit_reward_mean_std(self, iit_entity_list: list[iit_entity]) -> list[iit_entity]:
        if len(iit_entity_list) == 0:
            return iit_entity_list
        
        for entity in iit_entity_list: 
            try:
                entity.completion_concatenated_embedding = self.reduce_embedding(entity)
            except Exception as e:
                print(f"[Error] calculate[reduce_embedding]: {e}")
                entity.completion_concatenated_embedding = None

        filtered_iit_entity_list = list(filter(lambda x: x.has_concatenated_embedding(), iit_entity_list))
        if len(filtered_iit_entity_list) == 0:
            return iit_entity_list

        tpm_sbs: np.ndarray = self.build_tpm_list(filtered_iit_entity_list, self.get_config().reduced_dimension)
        for entity in filtered_iit_entity_list:
            try:
                weights_ts: np.ndarray = self.build_weights(entity)
                entity.iit_reward_raw = self.calculate_iit(entity, tpm_sbs, weights_ts)
                entity.iit_reward = self.compute_last_layer_on_reward(entity.iit_reward_raw)
                tpm_loss, tpm_entropy = self.calculate_tpm_loss_entropy(entity, tpm_sbs)
                entity.tpm_loss = tpm_loss
                entity.tpm_entropy = tpm_entropy
            except Exception as e:
                print(f"[Error] calculate[calculate_iit]: {e}")
                entity.iit_reward = 0.0
                entity.iit_reward_raw = 0.0
                entity.iit_reward_raw_actual = 0.0
            
            
        iit_entity.release_memory(iit_entity_list)
        return iit_entity_list

    def calculate_iit_reward_prompt(self, iit_entity_list: list[iit_entity]) -> iit_entity:
        if len(iit_entity_list) == 0:
            return None

        dimension = self.get_config().reduced_dimension
        for entity_tpm in iit_entity_list: 
            try:
                entity_tpm.completion_concatenated_embedding = self.reduce_embedding(entity_tpm)
            except Exception as e:
                print(f"[Error] calculate[reduce_embedding]: {e}")
                entity_tpm.completion_concatenated_embedding = None

        filtered_iit_entity_list = list(filter(lambda x: x.has_concatenated_embedding, iit_entity_list))
        if len(filtered_iit_entity_list) == 0:
            return None
        
        tpm_sbs: np.ndarray = self.build_tpm_list(filtered_iit_entity_list, dimension)
        for entity in filtered_iit_entity_list:
            try:
                weights_ts: np.ndarray = self.build_weights(entity)
                entity.iit_reward_raw = self.calculate_iit(entity, tpm_sbs, weights_ts)
                entity.iit_reward = self.compute_last_layer_on_reward(entity.iit_reward_raw)
                tpm_loss, tpm_entropy = self.calculate_tpm_loss_entropy(entity, tpm_sbs)
                entity.tpm_loss = tpm_loss
                entity.tpm_entropy = tpm_entropy
            except Exception as e:
                print(f"[Error] calculate[calculate_iit]: {e}")
                entity.iit_reward = 0.0
                entity.iit_reward_raw = 0.0
                entity.iit_reward_raw_actual = 0.0
        
        entity = min(filtered_iit_entity_list, key=lambda x: x.tpm_entropy)
        iit_entity.release_memory(iit_entity_list)
        return entity

    @abstractmethod
    def calculate_iit(tpm_sbs, weights_ts) -> float:
        pass

    def calculate_tpm_loss_entropy(self, entity : iit_entity, tpm_sbs) -> tuple[float, float]:
        markov_chain = entity.markov_chain
        tpm_loss, tpm_entropy = 0, 0
        for s1, s2 in zip(markov_chain, markov_chain[1:]):
            current_state = pyphi.convert.state2le_index(s1)
            next_state = pyphi.convert.state2le_index(s2)
            tpm_loss += -1 * math.log(tpm_sbs[current_state,next_state])
            tpm_entropy += -1 * tpm_sbs[current_state,next_state] * math.log(tpm_sbs[current_state,next_state])
        
        tpm_loss /= (len(markov_chain) -1)
        tpm_entropy /= (len(markov_chain) -1)
        return tpm_loss, tpm_entropy

    def refine_embedding(self, entity: iit_entity) -> iit_entity: 
        prompt_embedding = entity.prompt_embedding
        response_embedding = entity.completion_embedding
        attention_score = self.attention_score(prompt_embedding, response_embedding)
        attention_weight = self.attention_weight(attention_score)
        attended_response = np.matmul(attention_weight, prompt_embedding)
        for t_entity in entity.iit_token_list:
            t_entity.set_attended_embedding(attended_response[:, t_entity.token_number, :])

        response_concatenated = np.concatenate(tuple(attended_response), axis=1,)
        response_for_pca = response_concatenated
        entity.completion_embedding_for_pca = response_for_pca
        return entity

    def reduce_embedding(self, entity: iit_entity) -> torch.tensor: 
        if entity.completion_embedding_for_pca is None:
            raise Exception(f'Completion Embedding is Null')
        
        if entity.completion_embedding_for_pca.shape[0] < self.get_config().reduced_dimension:
            raise Exception(f'Embedding with {entity.completion_embedding_for_pca.shape} does not have the capability for dimensionality reduction to {self.get_config().reduced_dimension}')
        
        pca = decomposition.PCA(n_components = self.get_config().reduced_dimension, random_state = self.seed + 42 * 5,)
        pca.fit(entity.completion_embedding_for_pca)
        return pca.transform(entity.completion_embedding_for_pca)

    def compute_last_layer_on_reward(self, x: float) -> float:
        if last_layer_computation_type_enum.EXP == self.get_config().last_layer_computation_type: 
            return 1 - math.exp(-1 * self.get_config().last_layer_computation_param * x)

        if last_layer_computation_type_enum.TANH == self.get_config().last_layer_computation_type: 
            return math.tanh(self.get_config().last_layer_computation_param * x)

        if last_layer_computation_type_enum.IDENTITY == self.get_config().last_layer_computation_type: 
            return x
        
        return 0.0

    def attention_score(self, prompt, response, l_mask_spans=None, l_mask_context=None):
        """
        Calculate attention score between prompt and response tensors.

        Args:
        - prompt: numpy array of shape (n_l, n_1, D) - prompt embeddings
        - response: numpy array of shape (n_l, n_2, D) - response embeddings

        Returns:
        - attention_scores: numpy array of shape (n_l, n_2, n_1) - attention scores
        """
        # Get dimensions
        n_l, n_1, D = prompt.shape
        _, n_2, _ = response.shape

        # Calculate the dot product between response and prompt embeddings
        # The resulting shape will be (n_l, n_2, n_1)
        attention_scores = np.matmul(response, prompt.transpose(0, 2, 1))  # (n_l, n_2, n_1)

        # Scale the attention scores by sqrt(D)
        attention_scores = attention_scores / np.sqrt(D)

        if (l_mask_spans is not None) and (l_mask_context is not None):

            # Create a boolean mask for indices in l_mask_context
            mask_context = np.zeros(n_1, dtype=bool)
            mask_context[l_mask_context] = True  # Set context indices to True

            # Create a boolean mask for indices in l_mask_spans
            mask_span = np.zeros(n_1, dtype=bool)
            mask_span[l_mask_spans] = True  # Set span indices to True

            # Create a mask for indices that are neither in l_mask_spans nor in l_mask_context
            mask_out_of_both = ~mask_span & ~mask_context

            # Modify attention scores where the index is in l_mask_context
            attention_scores[:, :, mask_context] *= 0.6

            # Modify attention scores where the index is not in l_mask_spans or l_mask_context
            attention_scores[:, :, mask_out_of_both] *= 0.2

        # attention_scores = softmax(attention_scores, axis=-1)  # Softmax over the last axis (n_1 dimension)

        return attention_scores


    # Apply softmax to get the final attention scores for each layer
    def attention_weight(self, attention_scores, axis=-1):
        # Subtract the max value of x along the specified axis to prevent overflow
        exps = np.exp(attention_scores - np.max(attention_scores, axis=axis, keepdims=True))
        # Calculate the sum of the exponentials along the specified axis
        # Divide each exponential value by the sum of exponentials to get probabilities
        return exps / np.sum(exps, axis=axis, keepdims=True)
    
    def build_tpm_list(self, iit_entity_list: list[iit_entity], reduced_dim : int) -> np.ndarray:

        # This function is used to construct a state-by-state TPM.
        #
        # Inputs:
        # 1) list of time_series; array with dimensions n_time_points X n_regions
        #
        # Outputs:
        # 1) tpm:       array with dimensions n_states X n_states)
        #               each entry gives the sum_probabilities of transition between two states.

        # Obtain binary time-series based on mean signal threshold.
        time_series_list = list(map(lambda x: x.completion_concatenated_embedding, iit_entity_list))
        concatenated_time_series = np.concatenate(time_series_list, axis=0)
        
        if iit_threashold_type_enum.AVERAGE == self.get_config().threashold_type: 
            avgs = np.mean(concatenated_time_series, axis=0)
        elif iit_threashold_type_enum.MEDIAN == self.get_config().threashold_type: 
            avgs = np.median(concatenated_time_series, axis=0)
        
        tpm: np.ndarray = np.zeros((2**reduced_dim, 2**reduced_dim))
        for entity in iit_entity_list:
            time_series = entity.completion_concatenated_embedding
            time_series_copy = np.copy(time_series)

            for i in range(len(avgs)):
                time_series[np.where(time_series_copy[:, i] >= avgs[i]), i] = 1
                time_series[np.where(time_series_copy[:, i] < avgs[i]), i] = 0

            time_series = time_series.astype(int)
            markov_chain = time_series.tolist()
            entity.set_all_markov_chain(markov_chain)

            # Loop through all transitions and populate TPM.
            for s1, s2 in zip(markov_chain, markov_chain[1:]):
                i = pyphi.convert.state2le_index(s1)
                j = pyphi.convert.state2le_index(s2)
                tpm[i][j] += 1

        # Create array for transition counts.
        transitions_total = np.sum(tpm, axis=-1)

        # Normalize counts in TPM to obtain probabilities.
        for div in range(len(transitions_total)):
            if transitions_total[div] != 0.0:
                tpm[div, :] /= transitions_total[div]

        return np.copy(tpm)

    def build_weights(self, entity : iit_entity) -> np.ndarray:
        markov_chain = entity.markov_chain
        weights_ts = np.zeros((2 ** self.get_config().reduced_dimension))

        for s in markov_chain:
            i = pyphi.convert.state2le_index(s)
            weights_ts[i] += 1

        # Normalize weights with respect to time-series length:
        weights_ts /= len(markov_chain)
        return weights_ts

    def build_markov_chain_list(self, iit_entity_list : list[iit_entity]):
        time_series_list = list(map(lambda x: x.completion_concatenated_embedding, iit_entity_list))
        concatenated_time_series = np.concatenate(time_series_list, axis=0)
        avgs = np.mean(concatenated_time_series, axis=0)
        
        for entity in iit_entity_list:
            time_series = entity.completion_concatenated_embedding
            time_series_copy = np.copy(time_series)

            for i in range(len(avgs)):
                time_series[np.where(time_series_copy[:, i] >= avgs[i]), i] = 1
                time_series[np.where(time_series_copy[:, i] < avgs[i]), i] = 0

            time_series = time_series.astype(int)
            markov_chain = time_series.tolist()
            entity.set_all_markov_chain(markov_chain)


        return iit_entity_list

    def get_config(self): 
        return self.config

    def set_config(self, value): 
        self.config = value
