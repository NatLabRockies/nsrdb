"""A framework for handling source data in the NLR resource format: .h5 source
files with meta and time_index datasets, all data is (n_time, n_sites).
"""

import logging
import os

import numpy as np

from nsrdb.data_model.base_handler import AncillaryVarHandler
from nsrdb.file_handlers.file_system import NSRDBFileSystem as NSRDBfs

logger = logging.getLogger(__name__)


class NrelVar(AncillaryVarHandler):
    """Framework for NLR source data extraction."""

    def __init__(self, name, var_meta, date, **kwargs):
        """
        Parameters
        ----------
        name : str
            NSRDB var name.
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        date : datetime.date
            Single day to extract data for.
        """

        self._grid = None
        self._time_index = None
        self._row_mask = None
        super().__init__(name, var_meta=var_meta, date=date, **kwargs)

    @property
    def pattern(self):
        """Get the source file pattern which is sent to glob().

        Returns
        -------
        str | None
        """
        pat = super().pattern
        if pat is None:
            pat = os.path.join(self.source_dir, '*{}*'.format(self.file_set))

        return pat

    def pre_flight(self):
        """Perform pre-flight checks - source file check.

        Returns
        -------
        missing : str
            Look for the source file and return the string if not found.
            If nothing is missing, return an empty string.
        """

        missing = ''
        if not NSRDBfs(self.file).isfile():
            missing = self.file

        return missing

    @property
    def row_mask(self):
        """Get the boolean time_index (row) mask for the single day.

        Returns
        -------
        np.ndarray
        """

        if self._row_mask is None:
            with NSRDBfs(self.file, use_rex=True) as f:
                ti = f.time_index

            self._row_mask = (
                (ti.year == self._date.year)
                & (ti.month == self._date.month)
                & (ti.day == self._date.day)
            )

        return self._row_mask

    @property
    def time_index(self):
        """Get the NLR source file native time index masked for the
        day of interest.

        Returns
        -------
        pd.DatetimeIndex
        """

        if self._time_index is None:
            with NSRDBfs(self.file, use_rex=True) as f:
                self._time_index = f.time_index[self.row_mask]

        return self._time_index

    @property
    def source_data(self):
        """Get single variable data of shape (n_time, n_sites) from the
        NREL source file.

        Returns
        -------
        np.ndarray
        """

        with NSRDBfs(self.file, use_rex=True) as f:
            locs = np.where(self.row_mask)[0]
            row_slice = slice(locs[0], locs[-1] + 1)
            data = f[self.name, row_slice, :]

        return data

    @property
    def grid(self):
        """Return the NREL resource file coordinates with elevation.

        Returns
        -------
        pd.DataFrame
        """

        if self._grid is None:
            with NSRDBfs(self.file, use_rex=True) as f:
                self._grid = f.meta[['latitude', 'longitude', 'elevation']]

        return self._grid
