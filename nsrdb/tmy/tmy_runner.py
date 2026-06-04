"""NSRDB Typical Meteorological Year (TMY) runner code."""

import logging
import os
import shutil
from concurrent.futures import as_completed
from typing import ClassVar

import h5py
import numpy as np
from cloud_fs import FileSystem
from rex import init_logger
from rex.utilities.execution import SpawnProcessPool

from nsrdb.data_model.variable_factory import VarFactory
from nsrdb.file_handlers.outputs import Outputs
from nsrdb.file_handlers.resource import Resource

from .tmy import Tmy

logger = logging.getLogger(__name__)


class TmyRunner:
    """Class to handle running TMY, collecting outs, and writing to files."""

    WEIGHTS: ClassVar = {
        'TMY': {
            'max_air_temperature': 0.05,
            'min_air_temperature': 0.05,
            'mean_air_temperature': 0.1,
            'max_dew_point': 0.05,
            'min_dew_point': 0.05,
            'mean_dew_point': 0.1,
            'max_wind_speed': 0.05,
            'mean_wind_speed': 0.05,
            'sum_dni': 0.25,
            'sum_ghi': 0.25,
        },
        'TDY': {'sum_dni': 1.0},
        'TGY': {'sum_ghi': 1.0},
    }

    def __init__(
        self,
        nsrdb_base_fp,
        years,
        weights,
        sites_per_worker=100,
        n_nodes=1,
        node_index=0,
        site_slice=None,
        out_dir='/tmp/scratch/tmy/',
        fn_out='tmy.h5',
        supplemental_fp=None,
        var_meta=None,
    ):
        """
        Parameters
        ----------
        nsrdb_base_fp : str
            Base nsrdb filepath to retrieve annual files from. Must include
            a single {} format option for the year. Can include * for an
            NSRDB multi file source.
        years : iterable
            Iterable of years to include in the TMY calculation.
        weights : dict
            Lookup of {dset: weight} where dset is a variable h5 dset name
            and weight is a fractional TMY weighting. All weights must
            sum to 1.0
        sites_per_worker : int
            Number of sites to run at once (sites per core/worker).
        n_nodes : int
            Number of nodes being run.
        node_index : int
            Index of this node job.
        site_slice : slice
            Sites to consider in the GLOBAL TMY run. If multiple jobs are being
            run, the site slice should be the same for all jobs, and slices the
            full spatial extent meta data.
        out_dir : str
            Directory to dump temporary output files.
        fn_out : str
            Final output filename.
        supplemental_fp : None | dict
            Supplemental data base filepaths including {} for year for
            uncommon dataset inputs to the TMY calculation. For example:
            {'poa': '/projects/pxs/poa_h5_dir/poa_out_{}.h5'}
        var_meta : str
            CSV filepath containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        """

        logger.info('Initializing TMY runner for years: {}'.format(years))
        logger.info('TMY weights: {}'.format(weights))

        self._nsrdb_base_fp = nsrdb_base_fp
        self._years = years
        self._weights = weights

        self._sites_per_worker = sites_per_worker
        self._n_nodes = n_nodes
        self._node_index = node_index
        self._site_chunks = None
        self._site_chunks_index = None

        self._site_slice = (
            slice(None)
            if site_slice is None
            else slice(*site_slice)
            if isinstance(site_slice, list)
            else site_slice
        )
        self._meta = None
        self._dsets = None

        self._out_dir = out_dir
        self._fn_out = fn_out
        self._final_fpath = os.path.join(self._out_dir, self._fn_out)

        self._supplemental_fp = supplemental_fp
        self._var_meta = var_meta

        if not os.path.exists(self._out_dir):
            os.makedirs(self._out_dir)

        self._tmy_obj = Tmy(
            self._nsrdb_base_fp,
            self._years,
            self._weights,
            site_slice=slice(0, 1),
            supplemental_fp=supplemental_fp,
        )

        out = self._setup_job_chunks(
            self.meta,
            self._sites_per_worker,
            self._n_nodes,
            self._node_index,
            self._out_dir,
        )
        self._site_chunks, self._site_chunks_index, self._f_out_chunks = out

        logger.info('Node meta data is: \n{}'.format(self.meta))
        logger.info(
            'Node index {} with n_nodes {} running site chunks: '
            '{} ... {}'.format(
                node_index,
                n_nodes,
                str(self._site_chunks)[:100],
                str(self._site_chunks)[-100:],
            )
        )
        logger.info(
            'Node index {} with n_nodes {} running site chunks '
            'indices: {} ... {}'.format(
                node_index,
                n_nodes,
                str(self._site_chunks_index)[:100],
                str(self._site_chunks_index)[-100:],
            )
        )
        logger.info(
            'Node index {} with n_nodes {} running fout chunks: '
            '{} ... {}'.format(
                node_index,
                n_nodes,
                str(self._f_out_chunks)[:100],
                str(self._f_out_chunks)[-100:],
            )
        )

    @staticmethod
    def _setup_job_chunks(
        meta, sites_per_worker, n_nodes, node_index, out_dir
    ):
        """Setup chunks and file names for a multi-chunk multi-node job.

        Parameters
        ----------
        meta : pd.DataFrame
            FULL NSRDB meta data.
        sites_per_worker : int
            Number of sites to run at once (sites per core/worker).
        n_nodes : int
            Number of nodes being run.
        node_index : int
            Index of this node job (if a multi node job is being run).
        out_dir : str
            Directory to dump temporary output files.

        Returns
        -------
        site_chunks : list
             List of slices setting the site chunks to be run by this job.
        site_chunks_index : list
            List of integers setting the site chunk indices to be run by
            this job.
        f_out_chunks : dict
            Dictionary of file output paths keyed by the site chunk indices.
        """

        arr = meta.index.values
        tmp = np.array_split(arr, np.ceil(len(arr) / sites_per_worker))
        site_chunks = [slice(x.min(), x.max() + 1) for x in tmp]
        site_chunks_index = list(range(len(site_chunks)))

        site_chunks = np.array_split(np.array(site_chunks), n_nodes)[
            node_index
        ].tolist()
        site_chunks_index = np.array_split(
            np.array(site_chunks_index), n_nodes
        )[node_index].tolist()

        f_out_chunks = {}
        chunk_dir = os.path.join(out_dir, 'chunks/')
        if not os.path.exists(chunk_dir):
            os.makedirs(chunk_dir)
        for ichunk in site_chunks_index:
            f_out = os.path.join(chunk_dir, 'temp_out_{}.h5'.format(ichunk))
            f_out_chunks[ichunk] = f_out

        return site_chunks, site_chunks_index, f_out_chunks

    @property
    def meta(self):
        """Get the full NSRDB meta data."""
        if self._meta is None:
            fpath, Handler = self._tmy_obj._get_fpath('ghi', self._years[0])
            with FileSystem(fpath) as f, Handler(f) as res:
                self._meta = res.meta.iloc[self._site_slice, :]

        return self._meta

    @property
    def dsets(self):
        """Get the NSRDB datasets excluding meta and time index."""
        if self._dsets is None:
            fpath, Handler = self._tmy_obj._get_fpath('ghi', self._years[0])
            with FileSystem(fpath) as f, Handler(f) as res:
                self._dsets = []
                for d in res.dsets:
                    if res.shapes[d] == res.shape:
                        self._dsets.append(d)

            if self._supplemental_fp is not None:
                self._dsets += list(self._supplemental_fp.keys())

            self._dsets.append('tmy_year')
            self._dsets.append('tmy_year_short')
            self._dsets = list(set(self._dsets))

        return self._dsets

    @property
    def site_chunks(self):
        """Get a list of site chunk slices to parallelize across"""
        return self._site_chunks

    @staticmethod
    def get_dset_attrs(dsets, var_meta=None):
        """Get output file dataset attributes for a set of datasets.

        Parameters
        ----------
        dsets : list
            List of dataset / variable names.
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.

        Returns
        -------
        attrs : dict
            Dictionary of dataset attributes keyed by dset name.
        chunks : dict
            Dictionary of chunk tuples keyed by dset name.
        dtypes : dict
            dictionary of numpy datatypes keyed by dset name.
        """

        attrs = {}
        chunks = {}
        dtypes = {}

        for dset in dsets:
            var_obj = VarFactory.get_base_handler(dset, var_meta=var_meta)
            attrs[dset] = var_obj.attrs
            chunks[dset] = var_obj.chunks
            dtypes[dset] = var_obj.final_dtype

            if 'units' in attrs[dset]:
                attrs[dset]['psm_units'] = attrs[dset]['units']
            if 'scale_factor' in attrs[dset]:
                attrs[dset]['psm_scale_factor'] = attrs[dset]['scale_factor']

        return attrs, chunks, dtypes

    def _collect_chunk(self, f_out_chunk, site_slice, out):
        """Add single chunk to final output."""
        f_chunk_basename = os.path.basename(f_out_chunk)
        with Resource(f_out_chunk, unscale=True) as chunk:
            for dset in self.dsets:
                try:
                    data = chunk[dset]
                except Exception as e:
                    m = (
                        f'Could not read file dataset "{dset}" from file '
                        f'"{f_chunk_basename}". Received the following '
                        f'exception: \n{e}'
                    )
                    logger.exception(m)
                    raise e
                else:
                    out[dset, :, site_slice] = data
        return out

    def _collect(self, purge_chunks=False):
        """Collect all chunked files into the final fout."""

        status_file = os.path.join(self._out_dir, 'collect_status.txt')
        status = self._pre_collect(status_file)
        self._init_final_fout()

        with Outputs(self._final_fpath, mode='a', unscale=True) as out:
            for i, f_out_chunk in self._f_out_chunks.items():
                site_slice = self.site_chunks[i]
                f_chunk_basename = os.path.basename(f_out_chunk)

                msg = f'Skipping file, already collected: {f_chunk_basename}'
                if f_chunk_basename not in status:
                    out = self._collect_chunk(
                        f_out_chunk=f_out_chunk, site_slice=site_slice, out=out
                    )
                    with open(status_file, 'a') as f:
                        f.write('{}\n'.format(os.path.basename(f_out_chunk)))
                    msg = (
                        f'Finished collecting #{i + 1} out of '
                        f'{len(self._f_out_chunks)} for sites {site_slice} '
                        f'from file {f_chunk_basename}'
                    )
                logger.info(msg)

        if purge_chunks:
            chunk_dir = os.path.dirname(
                next(iter(self._f_out_chunks.values()))
            )
            logger.info('Purging chunk directory: {}'.format(chunk_dir))
            shutil.rmtree(chunk_dir)

    def _pre_collect(self, status_file):
        """Check to see if all chunked files exist before running collect

        Parameters
        ----------
        status_file : str
            Filepath to status file with a line for each file that has been
            collected.

        Returns
        -------
        status : list
            List of filenames that have already been collected.
        """
        missing = [
            fp for fp in self._f_out_chunks.values() if not os.path.exists(fp)
        ]
        if any(missing):
            emsg = 'Chunked file outputs are missing: {}'.format(missing)
            logger.error(emsg)
            raise FileNotFoundError(emsg)
        msg = 'All chunked files found. Running collection.'
        logger.info(msg)

        status = []
        if os.path.exists(status_file):
            with open(status_file) as f:
                status = f.readlines()
                status = [s.strip('\n') for s in status]
        return status

    def _init_final_fout(self):
        """Initialize the final output file."""
        self._init_file(
            self._final_fpath,
            self.dsets,
            self._tmy_obj.time_index,
            self.meta,
            var_meta=self._var_meta,
        )

    @staticmethod
    def _init_file(f_out, dsets, time_index, meta, var_meta=None):
        """Initialize an output file.

        Parameters
        ----------
        f_out : str
            File path to final .h5 file.
        dsets : list
            List of dataset names to initialize
        time_index : pd.datetimeindex
            Time index to init to file.
        meta : pd.DataFrame
            Meta data to init to file.
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        """

        if not os.path.isfile(f_out):
            dsets_mod = [d for d in dsets if 'tmy_year' not in d]
            attrs, chunks, dtypes = TmyRunner.get_dset_attrs(
                dsets_mod, var_meta=var_meta
            )
            dsets_mod.append('tmy_year')
            attrs['tmy_year'] = {
                'units': 'selected_year',
                'scale_factor': 1,
                'psm_units': 'selected_year',
                'psm_scale_factor': 1,
            }
            chunks['tmy_year'] = chunks['dni']
            dtypes['tmy_year'] = np.uint16
            Outputs.init_h5(
                f_out, dsets_mod, attrs, chunks, dtypes, time_index, meta
            )

            with h5py.File(f_out, mode='a') as f:
                d = 'tmy_year_short'
                f.create_dataset(d, shape=(12, len(meta)), dtype=np.uint16)
                f[d].attrs['units'] = 'selected_year'
                f[d].attrs['scale_factor'] = 1
                f[d].attrs['psm_units'] = 'selected_year'
                f[d].attrs['psm_scale_factor'] = 1

    @staticmethod
    def _write_output(f_out, data_dict, time_index, meta, var_meta=None):
        """Initialize and write an output file chunk.

        Parameters
        ----------
        f_out : str
            File path to final .h5 file.
        data_dict : dict
            {Dset: data_arr} dictionary
        time_index : pd.datetimeindex
            Time index to init to file.
        meta : pd.DataFrame
            Meta data to init to file.
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        """

        logger.debug('Saving TMY results to: {}'.format(f_out))
        tmp_file = f_out + '.tmp'
        TmyRunner._init_file(
            tmp_file,
            list(data_dict.keys()),
            time_index,
            meta,
            var_meta=var_meta,
        )
        with Outputs(tmp_file, mode='a') as f:
            for dset, arr in data_dict.items():
                f[dset] = arr
        os.rename(tmp_file, f_out)

    @staticmethod
    def _run_file(fp):
        """Check whether to run tmy for a given output filepath based on
        whether that file already exists and its file size."""
        run = True
        if os.path.exists(fp):
            size = os.path.getsize(fp)
            if size > 1e6:
                run = False

        return run

    @staticmethod
    def run_single(
        nsrdb_base_fp,
        years,
        weights,
        site_slice,
        dsets,
        f_out,
        supplemental_fp=None,
        var_meta=None,
    ):
        """Run TMY for a single site chunk (slice) and save to disk.

        Parameters
        ----------
        nsrdb_base_fp : str
            Base nsrdb filepath to retrieve annual files from. Must include
            a single {} format option for the year. Can include * for an
            NSRDB multi file source.
        years : iterable
            Iterable of years to include in the TMY calculation.
        weights : dict
            Lookup of {dset: weight} where dset is a variable h5 dset name
            and weight is a fractional TMY weighting. All weights must
            sum to 1.0
        site_slice : slice
            Sites to consider in this TMY chunk.
        dsets : list
            List of TMY datasets to make.
        f_out : str
            Filepath to save file for this chunk.
        supplemental_fp : None | dict
            Supplemental data base filepaths including {} for year for
            uncommon dataset inputs to the TMY calculation. For example:
            {'poa': '/projects/pxs/poa_h5_dir/poa_out_{}.h5'}
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.

        Returns
        -------
        True
        """

        run = TmyRunner._run_file(f_out)

        if run:
            data_dict = {}
            tmy = Tmy(
                nsrdb_base_fp=nsrdb_base_fp,
                years=years,
                weights=weights,
                site_slice=site_slice,
                supplemental_fp=supplemental_fp,
            )
            for dset in dsets:
                data_dict[dset] = tmy.get_tmy_timeseries(dset)
            TmyRunner._write_output(
                f_out, data_dict, tmy.time_index, tmy.meta, var_meta=var_meta
            )
        else:
            logger.info(
                'Skipping chunk, f_out already exists: {}'.format(f_out)
            )

        return True

    def _run_serial(self):
        """Run serial tmy futures and save temp chunks to disk."""

        logger.info(
            f'Running in serial for {len(self.site_chunks)} site chunks'
        )
        for i, site_slice in enumerate(self.site_chunks):
            self.run_single(
                nsrdb_base_fp=self._nsrdb_base_fp,
                years=self._years,
                weights=self._weights,
                site_slice=site_slice,
                dsets=self.dsets,
                f_out=self._f_out_chunks[self._site_chunks_index[i]],
                supplemental_fp=self._supplemental_fp,
                var_meta=self._var_meta,
            )

            logger.info(
                f'{i + 1} out of {len(self.site_chunks)} TMY chunks completed.'
            )

    def _run_parallel(self):
        """Run parallel tmy futures and save temp chunks to disk."""
        futures = {}
        loggers = ['nsrdb']

        logger.info(
            f'Running in parallel for {len(self.site_chunks)} site chunks'
        )
        with SpawnProcessPool(loggers=loggers) as exe:
            logger.info(f'Kicking off {len(self.site_chunks)} futures.')
            for i, site_slice in enumerate(self.site_chunks):
                future = exe.submit(
                    self.run_single,
                    nsrdb_base_fp=self._nsrdb_base_fp,
                    years=self._years,
                    weights=self._weights,
                    site_slice=site_slice,
                    dsets=self.dsets,
                    f_out=self._f_out_chunks[self._site_chunks_index[i]],
                    supplemental_fp=self._supplemental_fp,
                    var_meta=self._var_meta,
                )
                futures[future] = i

            logger.info(f'Finished kicking off {len(futures)} futures.')

            for i, future in enumerate(as_completed(futures)):
                if future.result():
                    logger.info(
                        f'{i + 1} out of {len(futures)} futures completed.'
                    )
                else:
                    logger.warning(f'Future #{i + 1} failed!')

    def _run(self):
        """Run in serial or parallel depending on number of chunks."""
        if len(self.site_chunks) > 1:
            self._run_parallel()
        else:
            self._run_serial()

    @classmethod
    def tmy(
        cls,
        nsrdb_base_fp,
        years,
        out_dir,
        fn_out,
        tmy_type='tmy',
        weights=None,
        sites_per_worker=100,
        n_nodes=1,
        node_index=0,
        log=True,
        log_level='INFO',
        log_file=None,
        site_slice=None,
        supplemental_fp=None,
        var_meta=None,
    ):
        """Run the TMY. Option for custom weights. Select default weights for
        TMY / TDY / TGY with `tmy_type` and `weights = None`"""

        if log:
            init_logger('nsrdb.tmy', log_level=log_level, log_file=log_file)

        if weights is None:
            weights = cls.WEIGHTS[tmy_type.upper()]

        tmy = cls(
            nsrdb_base_fp,
            years,
            weights,
            sites_per_worker=sites_per_worker,
            out_dir=out_dir,
            fn_out=fn_out,
            n_nodes=n_nodes,
            node_index=node_index,
            site_slice=site_slice,
            supplemental_fp=supplemental_fp,
            var_meta=var_meta,
        )
        tmy._run()

    @classmethod
    def collect(
        cls,
        nsrdb_base_fp,
        years,
        out_dir,
        fn_out,
        sites_per_worker=100,
        site_slice=None,
        supplemental_fp=None,
        var_meta=None,
        log=True,
        log_level='INFO',
        log_file=None,
        purge_chunks=False,
    ):
        """Run TMY collection.

        Parameters
        ----------
        nsrdb_base_fp : str
            Base nsrdb filepath to retrieve annual files from. Must include
            a single {} format option for the year. Can include * for an
            NSRDB multi file source.
        years : iterable
            Iterable of years to include in the TMY calculation.
        out_dir : str
            Directory to dump temporary output files.
        fn_out : str
            Final output filename.
        sites_per_worker : int
            Number of sites to run at once (sites per core/worker). Used here
            to determine size of chunks to collect. Needs to match the value
            used during the initial call to `tmy()`.
        site_slice : slice
            Sites to consider in this TMY.
        supplemental_fp : None | dict
            Supplemental data base filepaths including {} for year for
            uncommon dataset inputs to the TMY calculation. For example:
            {'poa': '/projects/pxs/poa_h5_dir/poa_out_{}.h5'}
        var_meta : str
            CSV filepath containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        log : bool
            Whether to write to logger
        log_level : str
            Log level for log output
        log_file : str | None
            Optional log file to write log output
        purge_chunks : bool
            Whether to purge chunks after collection into single file.
        """
        if log:
            init_logger('nsrdb.tmy', log_level=log_level, log_file=log_file)
        tgy = cls(
            nsrdb_base_fp,
            years,
            sites_per_worker=sites_per_worker,
            weights={'sum_ghi': 1.0},
            out_dir=out_dir,
            fn_out=fn_out,
            n_nodes=1,
            node_index=0,
            site_slice=site_slice,
            supplemental_fp=supplemental_fp,
            var_meta=var_meta,
        )
        tgy._collect(purge_chunks=purge_chunks)
