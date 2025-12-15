# -*- coding: utf-8 -*-
import numpy as np
from sklearn.utils import check_random_state

from .utils.gs_utils import get_distances
from . import growingspheres


class CounterfactualExplanation:
    """
    Class for defining a Counterfactual Explanation: this class will help point to specific counterfactual approaches
    """
    def __init__(self, obs_to_interprete, prediction_fn, method='GS', target_class=None, random_state=None):
        """
        Init function
        method: algorithm to use
        random_state
        """
        self.obs_to_interprete = obs_to_interprete
        self.prediction_fn = prediction_fn
        self.method = method
        self.target_class = target_class
        self.random_state = check_random_state(random_state)
        
        self.methods_ = {'GS': growingspheres.GrowingSpheres,
                         #'HCLS': lash.HCLS,
                         #'directed_gs': growingspheres.DirectedGrowingSpheres
                        }
        self.fitted = 0
        
    def fit(self, caps=None, n_in_layer=2000, layer_shape='ball', first_radius=0.1, dicrease_radius=10, num_enemies=1, sparse=True, verbose=False):
        """
        find the counterfactual with the specified method
        """
        # we should repeat this passage and checking if the number of enemies found is at least num_enemies
        found = 0
        self.enemies = []
        self.e_stars = []
        self.moves = []

        while found < num_enemies:
            cf = self.methods_[self.method](self.obs_to_interprete,
                self.prediction_fn,
                self.target_class,
                caps,
                n_in_layer,
                layer_shape,
                first_radius,
                dicrease_radius,
                sparse,
                verbose)
            enemies = cf.find_counterfactual(num_enemies=num_enemies)
            last_founds = len(cf.e_star) if isinstance(cf.e_star, (list, np.ndarray)) else 1
            found += last_founds
            
            self.enemies = self.enemies + list(enemies)
            self.e_stars = self.e_stars + list(cf.e_star)
            self.moves = self.moves + list(np.array(self.enemies[-last_founds:]) - self.obs_to_interprete)
        self.fitted = 1

    def distances(self, metrics=None):
        """
        scores de distances entre l'obs et le counterfactual
        """
        if self.fitted < 1:
            raise AttributeError('CounterfactualExplanation has to be fitted first!')
        return get_distances(self.obs_to_interprete, self.enemy)

    @property
    def enemy(self):
        """ for backward compatibility, return the first enemy only """
        return self.enemies[0] if self.enemies else None

    @property
    def e_star(self):
        """ for backward compatibility, return the first e_star only """
        return self.e_stars[0] if self.e_stars else None

    @property
    def move(self):
        """ for backward compatibility, return the first move only """
        return self.moves[0] if self.moves else None
