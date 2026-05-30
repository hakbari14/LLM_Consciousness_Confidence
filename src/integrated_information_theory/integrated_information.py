from src.integrated_information_theory.config.integrated_information_theory_config import integrated_information_theory_config
from src.integrated_information_theory.entity.iit_entity import iit_entity
from .integrated_information_theory import integrated_information_theory
from src.utils.enums_class import ii_phi_type_enum
from itertools import combinations

import numpy as np
import pyphi

pyphi.config.PROGRESS_BARS = False

class integrated_information(integrated_information_theory):

    def __init__(self, config: integrated_information_theory_config):
        super().__init__(config)


    def calculate_iit(self, entity: iit_entity, tpm_sbs: np.ndarray, weights_ts: np.ndarray) -> float:
        # This function computes the mean of phi_max for a time-series.
        # Inputs:
        # 1) tpm_sbs;   State-by-State form, dimensions n_states X n_states
        # 2) weights;   Array with dimensions 1 X n_states
        #
        # 3) subset_type;   Type of subset analysis
        #           "all" : Find phi_max for from all possible subsets
        #           "full": Find phi for the subset including all network elements
        #
        # Outputs:
        # 1) phi_weight
        #       Weighted average of phi_max for the TPM's corresponding time-series.
        tpm_sbn = pyphi.convert.to_2dimensional(pyphi.convert.state_by_state2state_by_node(tpm_sbs))

        rows, columns = tpm_sbn.shape
        setting_int = np.linspace(0, rows - 1, num=rows).astype(int)
        M = list(map(lambda x: list(pyphi.convert.le_index2state(x, columns)), setting_int))
        M = np.asarray(M).astype(int)

        phi_values = []
        phi_structures = []
        try:
            network = pyphi.Network(tpm_sbn)
        except:
            return 0.0, [0.0] * 16

        for state in range(rows):
            if weights_ts[state] != 0:

                substrate = network

                # Define candidate complex
                substrate_state = M[state, :]

                all_potential_mechanisms = self.all_subsets(network.size, thresh=-1)
                all_potential_mechanisms = dict(
                    [(x, 0) for x in all_potential_mechanisms]
                )
                candidate_complex_units = range(network.size)

                try:
                    candidate_complex = pyphi.Subsystem(substrate, substrate_state, nodes=candidate_complex_units)
                    dir(pyphi)
                    if ii_phi_type_enum.BIG_PHI == self.get_config().phi_type:
                        phi_structure_ = pyphi.new_big_phi.phi_structure(candidate_complex)
                except Exception as e:
                    print(e)
                    phi_values.append(0.0)
                    phi_structures.append([0.0] * 16)
                    continue

                if ii_phi_type_enum.BIG_PHI == self.get_config().phi_type:
                    phi_value = phi_structure_.big_phi
                elif ii_phi_type_enum.SYSTEM_PHI == self.get_config().phi_type:
                    phi_value = 0
                    for candidate_nodes in pyphi.utils.powerset(network.node_labels, nonempty=True):
                        try:
                            subsystem = pyphi.Subsystem(substrate, substrate_state, nodes=candidate_nodes)
                            sia = subsystem.sia()
                            phi_value = max(phi_value, sia.phi)
                        except pyphi.exceptions.StateUnreachableError:
                            phi_value = max(phi_value, 0.0)
                
                entity.set_integrated_information_value(state, phi_value)
                phi_values.append(phi_value)

        weights_non_zero = weights_ts[np.where(weights_ts != 0)]
        mean_phi = np.sum(phi_values * weights_non_zero)
        entity.iit_reward_raw = mean_phi
        return mean_phi

    def all_subsets(self, n, thresh=1):

        # This function is used to obtain all subsets in a full-subset analysis.
        # Inputs:
        # 1) Network size (n); number of regions in network

        # Outputs:
        # 1) subset_list; list of all possible subsets for given network size

        subset_list = []

        for n_i in range(n + 1):
            sub = list(combinations(range(n), n_i))

            if len(sub[0]) > thresh:
                for s in sub:
                    subset_list.append(s)

        return subset_list


