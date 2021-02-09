import random
from abc import ABC, abstractmethod
from itertools import permutations
from typing import List, Optional

from .config import DataConfig
from .sequence import State
from .special_tokens import SpecialTokens
from .template import get_template
from .token import Token, Segment, AnaType, GroupingOp
from .util import load_json


class Analysis(ABC):
    def __init__(self, ana_type: AnaType, aUID: str, config: DataConfig):
        self.type = ana_type
        self.aUID = aUID
        self.config = config

    def get_template(self):
        return get_template(self.type, self.config.allow_multiple_values, self.config.consider_grouping_operations)

    @abstractmethod
    def complete_states(self) -> List[State]:
        pass


class Chart(Analysis):
    __slots__ = "type", "aUID", "config", "x_fields", "values", "states", "grouping"

    def __init__(self, cUID: str, ana_type: AnaType, idx_to_field: dict,
                 config: DataConfig, search_sampling: bool):
        chart = load_json(config.chart_path(cUID), config.encoding)
        super().__init__(ana_type, cUID, config)

        self.values = [idx_to_field[field["index"]] for field in chart["values"]]
        if len(self.values) == 0:
            raise ValueError("No values!")
        if config.allow_multiple_values and len(self.values) > config.max_val_num:
            raise ValueError("Too many values in a chart.")

        cat_indices = [field["index"] for field in chart["categories"]]
        cat_indices.sort()
        for i in range(len(cat_indices) - 1):
            if cat_indices[i + 1] != cat_indices[i] + 1:
                raise ValueError("Category fields not continuous!")
        self.x_fields = [idx_to_field[index] for index in cat_indices]

        self.grouping = GroupingOp.from_raw_str(chart["grouping"]) if "grouping" in chart else None
        if ana_type is AnaType.BarChart and self.grouping is None:
            self.grouping = GroupingOp.Cluster

        # If using multiple values, we don't split the values.
        if config.allow_multiple_values:
            self.states = self.get_states(self.x_fields, self.values, self.grouping, search_sampling)
        else:
            self.states = []
            for value in self.values:
                value_states = self.get_states(self.x_fields, [value], self.grouping, search_sampling)
                self.states.extend(value_states)

    def get_states(self, x_fields: List[Token], values: List[Token],
                   grouping: Optional[GroupingOp], search_sampling: bool) -> List[State]:
        """
        :return: a list of all complete states
        """
        # Get all permutations of values and keep the permutation orders when deduplication.
        values_permutations = []
        values_set = set()
        for permutation in permutations(values):
            if permutation not in values_set:
                values_set.add(permutation)
                values_permutations.append(permutation)

        # If search_sampling, we use all value_permutations to train
        # else sample config.
        selected_states = list()
        if search_sampling:
            for values_perm in values_permutations:
                selected_states.append(self.get_state(x_fields, values_perm, grouping))
        else:
            # Get a fixed set of random indexes (by set a fixed random seed)
            num_permute_samples = min(self.config.num_permute_samples[len(values)], len(values_permutations))
            random.seed(1003)
            # Make sure the order of permutation (1st is the original order)
            selected_indexes = {0}  # Make sure that the permutation with the origin order is selected
            selected_indexes.update(random.sample(list(range(1, len(values_permutations))), k=num_permute_samples - 1))

            # Apply the random indexes to value_permutation.
            # An example:
            #   origin values are (1,2,3)
            #   permutations are (1,2,3), (1,3,2), (2,1,3), (2,3,1), (3,1,2), (3,2,1)
            #   selected_indexes are [4,1,3]
            #   Then we'll take perm[0,1,3], i.e. (1,2,3), (1,3,2), (2,3,1) into our dataset.
            for perm_idx in sorted(selected_indexes):
                values_perm = values_permutations[perm_idx]
                selected_states.append(self.get_state(x_fields, values_perm, grouping))

        return selected_states

    def get_state(self, x_fields: List, values: List, grouping: Optional[GroupingOp]) -> State:
        state = State.fill_template(self.get_template(), {
            Segment.X: x_fields,
            Segment.VAL: values,
            Segment.GRP: None if grouping is None else [SpecialTokens.get_grp_token(grouping)]
        })
        return state

    def seq_len(self) -> int:
        if not hasattr(self, "states") or len(self.states) == 0:
            return 0
        return len(self.states[0])

    def complete_states(self) -> List[State]:
        return self.states
