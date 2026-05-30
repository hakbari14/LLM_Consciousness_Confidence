from src.integrated_information_theory.config.integrated_information_theory_config import integrated_information_theory_config
from src.integrated_information_theory.entity.iit_entity import iit_entity
from .integrated_information_theory import integrated_information_theory
from ..utils.enums_class import ii_calculation_type_enum
import numpy as np
import math 
import pyphi


class intrinsic_information(integrated_information_theory):

    def __init__(self, config: integrated_information_theory_config):
        super().__init__(config)


    def calculate_iit(self, entity: iit_entity, tpm_sbs: np.ndarray, weights_ts: np.ndarray) -> float:
        entity.iit_reward_raw_actual = self.calculate_iit_actual(entity, tpm_sbs)
        return self.calculate_iit_maximal(entity, tpm_sbs, weights_ts)

    def calculate_iit_maximal(self, entity: iit_entity, tpm_sbs, weights_ts) -> float:

        # This function computes the mean of max of intrinsic information for a time-series.
        # Inputs:
        # 1) tpm_sbs;   State-by-State form, dimensions n_states X n_states
        # 2) weights;   Array with dimensions 1 X n_states
        #
        # Outputs:
        # 1) intrinsic information

        rows = tpm_sbs.shape[0]
        tbs_columns_average = tpm_sbs.mean(axis=0)
        ii_values = []
        
        for current_state in range(rows):
            if weights_ts[current_state] == 0:
                continue

            ii_effect_array = np.zeros((rows))
            for effect_state in range(rows): 
                p_constraint_effect = tpm_sbs[current_state, effect_state]
                if p_constraint_effect == 0: 
                    ii_effect_array[effect_state] = 0.0
                    continue

                p_unconstraint_effect = tbs_columns_average[effect_state]
                p_selectivity = p_constraint_effect

                ii_e = p_selectivity * math.log(p_constraint_effect / p_unconstraint_effect)
                ii_effect_array[effect_state] = ii_e
            
            ii_cause_array = np.zeros((rows))
            for cause_state in range(rows): 
                p_constraint_cause = tpm_sbs[cause_state, current_state]
                if p_constraint_cause == 0: 
                    ii_cause_array[cause_state] = 0.0
                    continue

                p_unconstraint_cause = tbs_columns_average[current_state]
                p_selectivity = p_constraint_cause / (tbs_columns_average[current_state] * rows)

                ii_cause = p_selectivity * math.log(p_constraint_cause / p_unconstraint_cause)
                ii_cause_array[cause_state] = ii_cause
            
            effect_state_index = np.argmax(ii_effect_array)
            max_effect = np.max(ii_effect_array)
            
            cause_state_index = np.argmax(ii_cause_array)
            max_cause = np.max(ii_cause_array)
            
            if ii_calculation_type_enum.MAX == self.get_config().calculation_type: 
                ii_value = max(max_effect, max_cause)
            elif ii_calculation_type_enum.SUM == self.get_config().calculation_type:
                ii_value = max_effect + max_cause
            
            entity.set_intrinsic_information_value(current_state, ii_value, max_effect, max_cause, effect_state_index, cause_state_index)
            ii_values.append(ii_value)


        weights_non_zero = weights_ts[np.where(weights_ts != 0)]
        mean_ii = np.sum(ii_values * weights_non_zero)

        return mean_ii

    def calculate_iit_actual(self, entity: iit_entity, tpm_sbs: np.ndarray) -> float :
        tbs_columns_average = tpm_sbs.mean(axis=0)
        rows = tpm_sbs.shape[0]
        markov_chain = entity.markov_chain
        total_ii_value = 0
        for idx, current_state in enumerate(markov_chain):
            current_state_index = pyphi.convert.state2le_index(current_state)

            effect_state_index = None
            if idx < len(markov_chain) - 1:
                effect_state_index = pyphi.convert.state2le_index(markov_chain[idx + 1])
                
            cause_state_index = None
            if idx > 0:
                cause_state_index = pyphi.convert.state2le_index(markov_chain[idx - 1])

            ii_effect = 0
            if effect_state_index is not None:
                p_constraint_effect = tpm_sbs[current_state_index, effect_state_index]
                if p_constraint_effect != 0: 
                    p_unconstraint_effect = tbs_columns_average[effect_state_index]
                    p_selectivity = p_constraint_effect
                    ii_effect = p_selectivity * math.log(p_constraint_effect / p_unconstraint_effect)

            ii_cause = 0
            if cause_state_index is not None:
                p_constraint_cause = tpm_sbs[cause_state_index, current_state_index]
                if p_constraint_cause != 0: 
                    p_unconstraint_cause = tbs_columns_average[current_state_index]
                    p_selectivity = p_constraint_cause / (tbs_columns_average[current_state_index] * rows)
                    ii_cause = p_selectivity * math.log(p_constraint_cause / p_unconstraint_cause)
            
            if ii_calculation_type_enum.MAX == self.get_config().calculation_type: 
                ii_value = max(ii_effect, ii_cause)
            elif ii_calculation_type_enum.SUM == self.get_config().calculation_type:
                ii_value = ii_effect + ii_cause
            total_ii_value += ii_value
        
        avg_ii_value = total_ii_value / (len(markov_chain) -1)
        return avg_ii_value
