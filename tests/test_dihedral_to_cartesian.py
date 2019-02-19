import encodermap as em
import numpy as np
from math import pi
import tensorflow as tf
import os
import tempfile
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


class TestDihedralToCartesianTf(tf.test.TestCase):
    def test_straight_to_helix(self):
        phi = (-57.8 / 180 - 1) * pi
        psi = (-47.0 / 180 - 1) * pi
        omega = 0.0
        dihedrals = [phi, psi, omega]*10
        result = [[0., 0., 0., ],
                  [1., 0., 0.],
                  [1.33166722, 0.94339645, 0.],
                  [0.96741215, 1.42302374, 0.79829563],
                  [1.08880021, 0.85355615, 1.61129723],
                  [0.72454514, 1.33318344, 2.40959286],
                  [-0.23997373, 1.54789329, 2.25596008],
                  [-0.70886876, 0.73651815, 1.90695029],
                  [-1.67338763, 0.951228, 1.75331751],
                  [-1.74499922, 1.71664339, 1.11377778],
                  [-1.14157263, 1.55412119, 0.33309675],
                  [-1.21318422, 2.31953658, -0.30644299],
                  [-0.97755207, 3.16914765, 0.16540288],
                  [-0.13573978, 3.03560281, 0.68839222],
                  [0.09989237, 3.88521388, 1.16023808],
                  [-0.65820911, 4.15648941, 1.75327411],
                  [-0.96694806, 3.3645823, 2.28011695],
                  [-1.72504954, 3.63585784, 2.87315298],
                  [-2.46428712, 4.01272662, 2.31503316],
                  [-2.63384221, 3.40789298, 1.53694105],
                  [-3.37307979, 3.78476176, 0.97882123],
                  [-3.12477419, 4.70531417, 0.6772792],
                  [-2.18945319, 4.69745201, 0.32356631],
                  [-1.94114759, 5.61800442, 0.02202428],
                  [-2.03578205, 6.25454616, 0.7874385],
                  [-1.60222551, 5.86370596, 1.59939456],
                  [-1.69685998, 6.50024771, 2.36480878],
                  [-2.66649702, 6.68630791, 2.52350853],
                  [-3.17315552, 5.82417585, 2.52855474],
                  [-4.14279257, 6.01023606, 2.68725449],
                  [-4.48234517, 6.62813877, 1.97809986],
                  [-4.17869445, 6.30364579, 1.08227588],
                  [-4.51824706, 6.92154852, 0.37312124]]

        with self.test_session():
            self.assertAllClose(result, em.dihedrals_to_cartesian_tf(dihedrals).eval(), atol=1e-4)

    def test_straight_to_helix_array(self):
        phi = (-57.8 / 180 - 1) * pi
        psi = (-47.0 / 180 - 1) * pi
        omega = 0.0
        dihedrals = tf.convert_to_tensor(np.array([[phi, psi, omega]*10]*10, dtype=np.float32))
        result = np.array([[[0., 0., 0., ],
                  [1., 0., 0.],
                  [1.33166722, 0.94339645, 0.],
                  [0.96741215, 1.42302374, 0.79829563],
                  [1.08880021, 0.85355615, 1.61129723],
                  [0.72454514, 1.33318344, 2.40959286],
                  [-0.23997373, 1.54789329, 2.25596008],
                  [-0.70886876, 0.73651815, 1.90695029],
                  [-1.67338763, 0.951228, 1.75331751],
                  [-1.74499922, 1.71664339, 1.11377778],
                  [-1.14157263, 1.55412119, 0.33309675],
                  [-1.21318422, 2.31953658, -0.30644299],
                  [-0.97755207, 3.16914765, 0.16540288],
                  [-0.13573978, 3.03560281, 0.68839222],
                  [0.09989237, 3.88521388, 1.16023808],
                  [-0.65820911, 4.15648941, 1.75327411],
                  [-0.96694806, 3.3645823, 2.28011695],
                  [-1.72504954, 3.63585784, 2.87315298],
                  [-2.46428712, 4.01272662, 2.31503316],
                  [-2.63384221, 3.40789298, 1.53694105],
                  [-3.37307979, 3.78476176, 0.97882123],
                  [-3.12477419, 4.70531417, 0.6772792],
                  [-2.18945319, 4.69745201, 0.32356631],
                  [-1.94114759, 5.61800442, 0.02202428],
                  [-2.03578205, 6.25454616, 0.7874385],
                  [-1.60222551, 5.86370596, 1.59939456],
                  [-1.69685998, 6.50024771, 2.36480878],
                  [-2.66649702, 6.68630791, 2.52350853],
                  [-3.17315552, 5.82417585, 2.52855474],
                  [-4.14279257, 6.01023606, 2.68725449],
                  [-4.48234517, 6.62813877, 1.97809986],
                  [-4.17869445, 6.30364579, 1.08227588],
                  [-4.51824706, 6.92154852, 0.37312124]]]*10, dtype=np.float32)

        with self.test_session():
            self.assertAllClose(result, em.dihedrals_to_cartesian_tf(dihedrals).eval(), atol=1e-4)

    def test_learn_helix(self):
        phi = (-57.8 / 180 - 1) * pi
        psi = (-47.0 / 180 - 1) * pi
        omega = 0.0
        dihedrals = [phi, psi, omega]*10
        with self.test_session():
            cartesian = em.dihedrals_to_cartesian_tf(dihedrals).eval()

        with tempfile.TemporaryDirectory() as temp_path:
            parameters = em.Parameters()
            parameters.main_path = temp_path
            parameters.dihedral_to_cartesian_cost_scale = 1
            parameters.auto_cost_scale = 0
            parameters.distance_cost_scale = 0
            parameters.l2_reg_constant = 0.
            parameters.batch_size = 1
            parameters.n_steps = 1000
            parameters.summary_step = 1

            e_map = em.EncoderMap(parameters, (np.expand_dims(dihedrals, 0), np.expand_dims(cartesian, 0)))
            e_map.train()

            latent = e_map.encode(np.expand_dims(dihedrals, 0))
            generated = e_map.generate(latent)

            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(*cartesian.T)
            ax.plot(*e_map.cartesians[-1][0].T)
            set_axes_equal(ax)
            plt.show()

    def test_straight_tetrahedral_chain_with_bond_lenght(self):
        result = [[0.       , 0.       , 0.       ],
                  [1.       , 0.       , 0.       ],
                  [1.6633345, 1.8867929, 0.       ],
                  [4.6633344, 1.8867929, 0.       ],
                  [4.995002 , 2.8301892, 0.       ],
                  [6.995002 , 2.8301892, 0.       ],
                  [7.990003 , 5.6603785, 0.       ]]
        cartesian = em.straight_tetrahedral_chain(bond_lengths=[1, 2, 3, 1, 2, 3])
        self.assertAllClose(result, cartesian)

        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot(*cartesian.T)
        # set_axes_equal(ax)
        # plt.show()
