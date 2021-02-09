from abc import ABC, abstractmethod
from copy import copy
from typing import Optional, List, Iterable

from Table2Charts.data import QValue, DataTable, Result


class Agent(ABC):
    @abstractmethod
    def done(self) -> bool:
        """
        :return: If the agent has done searching.
        """
        pass

    @abstractmethod
    def table(self) -> DataTable:
        """
        :return: On which table the agent is working.
        """
        pass

    @abstractmethod
    def ranked_complete_states(self) -> List[Result]:
        """
        :return: Current complete results.
        """
        pass

    @abstractmethod
    def step(self) -> List[QValue]:
        """
        Take a step forward: (Initialize and) Search the states for estimation.
        :return: Chosen (state, actions) pairs for model prediction/scoring.
        """
        pass

    @staticmethod
    def valid_results(chosen: List[QValue], predicted_values: Iterable) -> Iterable[Result]:
        for state_actions, action_values in zip(chosen, predicted_values):
            state = state_actions.state
            actions = state_actions.actions
            valid_mask = state_actions.valid_mask

            for action, valid, score in zip(actions, valid_mask, action_values[:len(actions)]):
                if not valid:
                    continue
                result = Result(score, copy(state).append(action))
                yield result

    @abstractmethod
    def update(self, chosen: List[QValue], predicted_values: Iterable) -> Optional[dict]:
        """
        Consume the estimation results.
        :param chosen: The chosen pairs returned by step().
        :param predicted_values: Predicted action values for the chosen pairs.
        :return: Recorder summary dict when the agent is done, else None.
        """
        pass
