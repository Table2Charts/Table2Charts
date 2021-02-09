from typing import Optional

import numpy as np

from Table2Charts.data import AnaType


class SearchConfig:
    def __init__(self, load_ground_truth: bool, search_all_types: bool = False,
                 dim_count: int = 600, com_count: int = 1100, max_rc: tuple = (4, 2),
                 expand_limit: int = 200, time_limit: float = np.inf,
                 frontier_size: int = 300, beam_size: int = 8, min_threshold: float = 0.,
                 log_path: Optional[str] = None, search_single_type: AnaType = None,
                 test_field_selections: bool = False, test_design_choices: bool = False):
        """
        :param load_ground_truth:
        :param dim_count:  max number of dim(fields)
        :param com_count:  max number of steps for com task
        :param max_rc:
        :param expand_limit:
        :param time_limit:
        :param frontier_size:
        :param beam_size:
        :param min_threshold:
        :param log_path:
        :param search_single_type: limit searching to the specified ana_type.
        If it is None, this limitation will not be applied.
        :param test_field_selections: whether to only test field selection
        :param test_design_choices: whether to only test charting (visualization)
        """
        self.load_ground_truth = load_ground_truth
        self.search_all_types = search_all_types
        self.search_single_type = search_single_type

        self.test_field_selections = test_field_selections
        self.test_design_choices = test_design_choices

        self.max_rc = max_rc
        self.expand_limit = expand_limit
        self.time_limit = time_limit
        self.dim_count = dim_count
        self.com_count = com_count

        self.frontier_size = frontier_size
        self.beam_size = beam_size
        self.min_threshold = min_threshold

        self.log_path = log_path
