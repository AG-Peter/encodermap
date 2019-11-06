from .autoencoder import Autoencoder
import tensorflow as tf
from .misc import distance_cost


class EncoderMap(Autoencoder):
    def _setup_cost(self):
        self._auto_cost()
        self._center_cost()
        self._l2_reg_cost()
        self._distance_cost()

    def _distance_cost(self):
        if self.p.distance_cost_scale is not None:
            dist_cost = distance_cost(self.main_inputs, self.latent, *self.p.dist_sig_parameters, self.p.periodicity)
            tf.summary.scalar("distance_cost", dist_cost)
            if self.p.distance_cost_scale != 0:
                self.cost += self.p.distance_cost_scale * dist_cost
