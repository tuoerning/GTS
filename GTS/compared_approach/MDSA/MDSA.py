
#Weiss et al. Simple techniques work surprisingly well for neural network test prioritization and active learning (replicability study)
#https://github.com/testingautomated-usi/simple-tip?tab=readme-ov-file

"""Prioritizers, such as Coverage-Total Method and Coverage-Additional Method"""
from typing import Generator
import time
import numpy as np


def ctm(scores: np.ndarray) -> Generator[int, None, None]:
    """Indexes according to Coverage-Total Method"""
    assert len(scores.shape) == 1
    # Sort negative scores to achieve decreasing order
    idxs = np.argsort(-scores)
    for x in idxs:
        yield x


def cam(scores: np.ndarray, profiles: np.ndarray) -> Generator[int, None, None]:
    """Indexes according to Coverage-Total Method (i.e., greedily increasing overall coverage)"""
    scores = scores.copy()
    profiles = profiles.reshape((profiles.shape[0], -1))
    uncovered = np.ones_like(profiles[0])
    num_coverable = np.sum(profiles, axis=1).flatten()
    remaining = np.sum(uncovered)
    yielded = np.zeros_like(scores)
    while True:
        next = np.argmax(num_coverable)
        covering_columns = profiles[next].nonzero()[0]
        newly_covered = num_coverable[next]

        if newly_covered == 0:
            break

        yield next
        yielded[next] = 1

        # Update uncovered
        remaining -= newly_covered  # Faster than np.sum(uncovered) after updating
        num_coverable_deductions = np.sum(profiles[:, covering_columns], axis=1)
        num_coverable = num_coverable - num_coverable_deductions
        uncovered[covering_columns] = 0
        profiles[:, covering_columns] = 0

        if remaining == 0:
            break

    # Sort remaining according to original scores and return
    min_score = np.min(scores) - 1
    # Make sure already yealed samples have a very low score and are at the and of the ordering
    scores[yielded.nonzero()[0]] = min_score - 1
    idxs = np.argsort(-scores)
    for x in idxs:
        # a score < min_score stands for a sample that was already yielded
        #   (and so will all further ones), so we end the loop
        if scores[x] < min_score:
            break
        else:
            yield x
            yielded[x] = True  # Unneeded, just for assertion below

    assert np.all(yielded)
    
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from surprise import DSA, LSA, MDSA, MLSA, MultiModalSA, SurpriseCoverageMapper


NUM_SC_BUCKETS = 1000

import tensorflow as tf


class SurpriseHandler:
    """Effenciently handles Surprise Adequacy instances."""

    TESTED_SA = {
        #Plain Distance-Based Surprise Adequacy
        "dsa": lambda x, y: DSA(x, y, subsampling=0.3),
        #Per-Class Likelihood Surprise Adequacy
        # "pc-lsa": lambda x, y: MultiModalSA.build_by_class(x, y, lambda x, y: LSA(x)),
        # # Per-Class  Mahalanobis Distance based Surprise Adequacy
        "pc-mdsa": lambda x, y: MultiModalSA.build_by_class(x, y, lambda x, y: MDSA(x)),
        #Per-Class  Multimodal Likelihood Surprise Adequacy
        "pc-mlsa": lambda x, y: MultiModalSA.build_by_class(
            x, y, lambda x, y: MLSA(x, num_components=3)
        ),
        # Per-Class  Multimodal Mahalanobis Distance based Surprise Adequacy
        "pc-mmdsa": lambda x, y: MultiModalSA.build_with_kmeans(
            x, y, lambda x, y: MDSA(x), potential_k=range(2, 6), subsampling=0.3
        ),
    }
    
    def _acti_and_pred(
        self, dataset: Union[np.ndarray, tf.data.Dataset]
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Collects activations and predicions in a single NN forward pass"""
        outputs = self.base_model.get_activations(dataset)
        assert len(outputs) == len(self.sa_layers) + 1
        return (outputs[:-1], np.argmax(outputs[-1], axis=1))

    def evaluate_all(self, train_ats, train_pred,test_ats,test_pred, dsa_badge_size: Optional[int] = None):
         
        """Collect all the different surprise adequacies for the passed datasets"""
        res = dict()
        # ats, predictions, times
        test_apt = dict()
        # Calc SAs
        for sa_name, sa_func in tqdm(self.TESTED_SA.items(), desc="Calculating SAs"):
            #s = time.time()
            res[sa_name] = dict()
            
            
            print(f"Creating {sa_name} instance")
            sa = sa_func(train_ats, train_pred)
            if isinstance(sa, DSA) and dsa_badge_size is not None:
                sa.badge_size = dsa_badge_size
          
            sa_pred = sa(test_ats, test_pred)
            res[sa_name] = (sa_pred)
            #e = time.time()
            #print(sa_name+"1:"+str(e-s))
            

        # Calc CAMs
        # We're interating over the keys of the inputs,
        #   instead of the items of res, to avoid modifying the dict which
        #   is being iterated over.
        for sa_name in self.TESTED_SA.keys():
                #s = time.time()
                sa_pred= res[sa_name]
               
                    # We use the max of all observed values to dynamically select the
                    #   buckets upper bound.
                coverage_mapper = SurpriseCoverageMapper(
                        NUM_SC_BUCKETS, np.max(sa_pred)
                )
                coverage_profiles = coverage_mapper.get_coverage_profile(sa_pred)
                cam_order = [i for i in cam(sa_pred, coverage_profiles)]
                cam_order = np.array(cam_order)
                res[sa_name] = (sa_pred, cam_order)
                #e = time.time()
                #print(sa_name+"2:"+str(e-s))



        
        return res