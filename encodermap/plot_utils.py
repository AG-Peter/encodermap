import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


class ProjectedPlotUtils(object):

    """

        Class with functions for plotting and calculating starting from the encoded data of Encodermap, saved in the
        projected.npy file

        Attributes
        ----------

            projected = 2D numpy array of EncoderMap encoded points.

        Methods
        --------

            kde_encodermap:
                Plot KDE of EncoderMap encoded data.

            covered_area:
                Calculate the total covered area of an EncoderMap plot as the total number of non-empty bins.

            hist_plot_projected:
                Plot the projected EncoderMap points as a 2D histogram.

            minima_hist_projected:
                Obtain position and number of minima (bins with value < threshold) in a certain interval
                of the EncoderMap plot, and plot them.

            minima_hist:
                Obtain position and number of minima (bins with value < threshold) in a defined interval
                of the EncoderMap plot.

            obtain_minima_frames:
                Obtain the trajectory frames corresponding to minimum values in the histogram2D plot.


    """

    def __init__(self, projected):

        self.projected = projected

    def kde_encodermap(self, figname='kde_encodermap', scatter_data=[], save=True, scatter=False):

        """

        Plot KDE of EncoderMap encoded data.

        :param projected: 2d numpy array of EncoderMap encoded points
        :param figname: name for the image .png file
        :param scatter_data: list of x and y coordinates of points to be represented on a scatterplot
        :param save: bool, choice about saving the image
        :param scatter: bool, to be used when plotting minima on the 2D map

        """

        # Create DataFrame from projected array
        projected_df = pd.DataFrame({'x': self.projected[0:, 0], 'y': self.projected[0:, 1]})

        sns.set_style('darkgrid')
        fig, ax = plt.subplots()

        # Plot KDE plot of Encodermap data
        sns.kdeplot(data=projected_df, x='x', y='y', fill=True,
                    bw_adjust=0.2, legend=True)

        if scatter == True:
            plt.scatter(scatter_data[0], scatter_data[1], marker='x', s=50, c='red')

        # Parameters setting for plot
        ax.set_title("")
        ax.set_xlabel('x', fontsize=20)
        ax.set_ylabel('y', fontsize=20)

        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(18)

        plt.show()

        # Save image as png
        if save:
            fig.savefig("{}.png".format(figname), dpi=320)

    def covered_area(self, interval, bins):

        """

        Calculate the total covered area of an EncoderMap plot as the total number of
        non-empty bins.

        :param projected: 2d numpy array of EncoderMap encoded points
        :param interval: [[xmin, xmax], [ymin, ymax]] interval for covered area calculation
        :param bins: number of bins for the 2D histogram

        :returns: area_covered: number of non-empty bins in interval

        """

        # Create 2D histogram for the points within the specified interval
        projected_interval = self.projected[(self.projected[:, 0] > interval[0][0]) &
                                            (self.projected[:, 0] < interval[0][1]) &
                                            (self.projected[:, 1] > interval[1][0]) &
                                            (self.projected[:, 1] < interval[1][1])]

        hist, xedges, yedges = np.histogram2d(projected_interval[:, 0],
                                              projected_interval[:, 1],
                                              bins=bins)

        # Calculate the covered area
        area_covered = sum(sum(hist != 0))

        return area_covered

    def hist_plot_projected(self, interval, bins, figname='hist2D_encodermap', save=True):

        """

        Plot the projected EncoderMap points as a 2D histogram.

        :param projected: 2d numpy array of EncoderMap encoded points
        :param interval: [[xmin, xmax], [ymin, ymax]] interval the calculation
        :param bins: number of bins for the 2D histogram
        :param figname: name for the image .png file to create
        :param save: bool, choice about saving the image

        """

        # Create 2D histogram for the points within the specified interval
        projected_interval = self.projected[(self.projected[:, 0] > interval[0][0]) &
                                            (self.projected[:, 0] < interval[0][1]) &
                                            (self.projected[:, 1] > interval[1][0]) &
                                            (self.projected[:, 1] < interval[1][1])]

        hist, xedges, yedges = np.histogram2d(projected_interval[:, 0],
                                              projected_interval[:, 1],
                                              bins=bins)

        # Plot the 2D histogram with values = -log(bin_count)
        fig, ax = plt.subplots()

        caxe = ax.imshow(-np.log(hist.T), origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                         aspect="auto")

        cbar = fig.colorbar(caxe)

        cbar.set_label("-ln(p)", labelpad=0)

        plt.show()

        # Save image as png
        if save:
            fig.savefig("{}.png".format(figname), dpi=320)

    def minima_hist_projected(self, interval, threshold, bins, figname='minima_encodermap', save=True):

        """

        Obtain position and number of minima (bins with value < threshold) in a certain interval
        of the EncoderMap plot, and plot them.

        :param projected: 2d numpy array of EncoderMap encoded points
        :param interval: [[xmin, xmax], [ymin, ymax]] interval the calculation
        :param threshold: value below which points are considered minima
        :param bins: number of bins for the 2D histogram
        :param figname: name for the image .png file to create
        :param save: bool, choice about saving the image

        """

        # Create 2D histogram for the points within the specified interval
        projected_interval = self.projected[(self.projected[:, 0] > interval[0][0]) &
                                            (self.projected[:, 0] < interval[0][1]) &
                                            (self.projected[:, 1] > interval[1][0]) &
                                            (self.projected[:, 1] < interval[1][1])]

        hist, xedges, yedges = np.histogram2d(projected_interval[:, 0],
                                              projected_interval[:, 1],
                                              bins=bins)

        log_hist = -np.log(hist)

        # Obtain the minima coordinates as the center of their bins, and their value
        x, y, lnp = [], [], []

        for i in range(0, bins):
            for j in range(0, bins):
                if log_hist[i, j] <= threshold:
                    x.append((xedges[i] + xedges[i + 1]) / 2)
                    y.append((yedges[j] + yedges[j + 1]) / 2)
                    lnp.append(log_hist[i, j])

        print(f'Number of minima below threshold={threshold}: {len(x)}')

        # Create DataFrames to plot the minima data
        minima = pd.DataFrame({'X': x, 'Y': y, '-ln(P)': lnp}).sort_values(by='-ln(P)', ignore_index=True)

        # Plot the the minima on the 2D map
        self.kde_encodermap(self.projected, figname, scatter_data=[x, y], scatter=True, save=False)

        # Save image as png
        if save:
            plt.savefig("{}.png".format(figname), dpi=320)

    def minima_hist(self, interval, threshold, bins):

        """

        Obtain position and number of minima (bins with value < threshold) in a defined interval
        of the EncoderMap plot.

        :param projected: 2d numpy array of EncoderMap encoded points
        :param interval: [[xmin, xmax], [ymin, ymax]] interval the calculation
        :param threshold: value below which points are considered minima
        :param bins: number of bins for the 2D histogram

        :returns: minima: DataFrame with X, Y coordinates and -ln(P) values of the minima

        """

        # Create 2D histogram for the points within the specified interval
        projected_interval = self.projected[(self.projected[:, 0] > interval[0][0]) &
                                            (self.projected[:, 0] < interval[0][1]) &
                                            (self.projected[:, 1] > interval[1][0]) &
                                            (self.projected[:, 1] < interval[1][1])]

        hist, xedges, yedges = np.histogram2d(projected_interval[:, 0],
                                              projected_interval[:, 1],
                                              bins=bins)

        log_hist = -np.log(hist)

        # Obtain the minima coordinates as the center of their bins, and their value
        x, y, lnp = [], [], []

        for i in range(0, bins):
            for j in range(0, bins):
                if log_hist[i, j] <= threshold:
                    x.append((xedges[i] + xedges[i + 1]) / 2)
                    y.append((yedges[j] + yedges[j + 1]) / 2)
                    lnp.append(log_hist[i, j])

        print(f'Number of minima below threshold={threshold}: {len(x)}')

        # Create DataFrames to plot the minima data
        minima = pd.DataFrame({'X': x, 'Y': y, '-ln(P)': lnp}).sort_values(by='-ln(P)', ignore_index=True)

        return minima

    def obtain_minima_frames(self, interval, threshold, bins, num_minima, filename='frames_minima', save=True):

        """

        Obtain the trajectory frames corresponding to minimum values in the histogram2D plot.

        :param projected: 2d numpy array of EncoderMap encoded points
        :param interval: [[xmin, xmax], [ymin, ymax]] interval the calculation
        :param threshold: value below which points are considered minima
        :param bins: number of bins for the 2D histogram
        :param num_minima: number of minima to consider
        :param filename: name for the .csv file to create
        :param save: bool, choice about saving the data to .csv file

        :returns frame_minima: list of trajectory frames corresponding to minima

        """

        minima = self.minima_hist(self.projected, interval, threshold, bins)

        # Obtain the X and Y coordinates of the first num_minima number of minima
        X = np.array(minima['X'].iloc[0:num_minima])
        Y = np.array(minima['Y'].iloc[0:num_minima])

        # Obtain the two columns of the projected points
        x = np.array(self.projected[:, 0])
        y = np.array(self.projected[:, 1])

        # Obtain the frames corresponding to each minima as those with minimum distance from
        # projected data
        frames_minima = []

        for i in range(0, num_minima):
            frames_minima.append(cdist([[X[i], Y[i]]], np.array([x, y]).T,
                                       metric='euclidean').argmin() + 1)

        # Save frames as csv file
        if save:
            pd.DataFrame(frames_minima).to_csv(f"{filename}.csv", header=None, index=None, sep= ',')

        return frames_minima


