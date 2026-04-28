"""Config creation class for CONUS / Full Disc NSRDB runs.

TODO: This can probably be streamlined a bunch more. Lots of hardcoded stuff
"""

import calendar
import copy
import json
import logging
import os
import pprint

from mlclouds import LEG_MODEL_FPATH

from nsrdb.aggregation.aggregation import NSRDB_2018
from nsrdb.config import (
    PIPELINE_CONFIG_TEMPLATE,
    POST2017_CONFIG_TEMPLATE,
    PRE2018_CONFIG_TEMPLATE,
)
from nsrdb.utilities.file_utils import get_format_keys, str_replace_dict

DEFAULT_EXEC_CONFIG = {
    'option': 'kestrel',
    'memory': 173,
    'walltime': 10,
    'alloc': 'pxs',
}

DEFAULT_META_DIR = '/projects/pxs/reference_grids/'

BASE_KWARGS = {
    'basename': 'nsrdb',
    'out_dir': './',
    'execution_control': DEFAULT_EXEC_CONFIG,
    'meta_dir': DEFAULT_META_DIR,
}

DEFAULT_MLCLOUDS = {
    'model_path': {'cloud_prop_model_path': LEG_MODEL_FPATH},
    'col_chunk': 10000,
    'fill_all': False,
    'max_workers': 4,
}

MAIN_KWARGS = {
    **BASE_KWARGS,
    'extent': 'full',
    'satellite': 'east',
    'ml-cloud-fill': DEFAULT_MLCLOUDS,
}

SURFRAD_KWARGS = {
    **MAIN_KWARGS,
    'freq': '15min',
    'spatial': '4km',
}

BLEND_KWARGS = {
    **BASE_KWARGS,
    'file_tag': 'all',
    'extent': 'full',
    'main_dir': '../',
}

COLLECT_BLEND_KWARGS = {**BASE_KWARGS, 'extent': 'full', 'max_workers': 1}

AGG_KWARGS = {
    **BASE_KWARGS,
    'full_spatial': '2km',
    'conus_spatial': '2km',
    'final_spatial': '4km',
    'full_freq': '10min',
    'conus_freq': '5min',
    'final_freq': '30min',
    'n_chunks': 32,
    'source_priority': ['conus', 'full_disc'],
}

COLLECT_AGG_KWARGS = {
    **BASE_KWARGS,
    'final_spatial': '4km',
    'final_freq': '30min',
    'max_workers': 10,
}

EXTENT_MAP = {
    'lon_seam': {'full': -105, 'conus': -113},
    'extent_tag': {'full': 'RadF', 'conus': 'RadC'},
    'map_col': {'full': 'gid_full', 'conus': 'gid_full_conus'},
}


DEFAULT_RES = {
    'POST_2018': {
        'conus': {'spatial': '2km', 'freq': '5min'},
        'full': {'spatial': '2km', 'freq': '10min'},
    },
    '2018': {
        'conus': {'spatial': '2km', 'freq': '5min'},
        'full': {
            'east': {'spatial': '2km', 'freq': '10min'},
            'west': {'spatial': '4km', 'freq': '30min'},
        },
    },
    'PRE_2018': {'spatial': '4km', 'freq': '30min'},
}

logger = logging.getLogger(__name__)


class CreateConfigs:
    """Collection of methods to create config files for NSRDB module inputs for
    standard CONUS / Full Disc runs."""

    MAIN_RUN_NAME = '{basename}_{satellite}_{extent}_{year}_{spatial}_{freq}'
    SURFRAD_RUN_NAME = '{basename}_{year}_surfrad'
    BLEND_RUN_NAME = '{basename}_{extent}_{year}_blend'
    AGG_RUN_NAME = '{basename}_{year}_aggregate'
    COLLECT_AGG_RUN_NAME = '{basename}_{year}_collect_aggregate'
    COLLECT_BLEND_RUN_NAME = '{basename}_{extent}_{year}_collect_blend'

    @classmethod
    def init_kwargs(cls, kwargs=None, default_kwargs=None):
        """Initialize config with default kwargs."""
        default_kwargs = default_kwargs or {}
        msg = f'kwargs must have a "year" key. Received {kwargs}.'
        assert 'year' in kwargs, msg
        config = copy.deepcopy(default_kwargs)
        input_kwargs = copy.deepcopy(kwargs)
        if 'execution_control' in kwargs:
            config['execution_control'].update(
                input_kwargs.pop('execution_control')
            )
        config.update(input_kwargs)
        for k, v in list(config.items()):
            if k.endswith('_dir'):
                config[k] = os.path.abspath(v)
        os.makedirs(config['out_dir'], exist_ok=True)
        return config

    @classmethod
    def _get_res(cls, config):
        """Get spatiotemporal res for a given year and extent."""

        spatial, freq = config.get('spatial', None), config.get('freq', None)
        if spatial is not None and freq is not None:
            return spatial, freq

        required_args = (
            ['extent', 'satellite']
            if config['year'] == 2018
            else ['extent']
            if config['year'] > 2018
            else []
        )
        msg = (
            'To automatically get resolution we need required_args: '
            f'{required_args}". Either provide "spatial" and "freq" or '
            'the required args.'
        )
        assert all(arg in config for arg in required_args), msg

        if config['year'] == 2018:
            res = DEFAULT_RES['2018'][config['extent']][config['satellite']]

        elif config['year'] > 2018:
            res = DEFAULT_RES['POST_2018'][config['extent']]
        else:
            res = DEFAULT_RES['PRE_2018']
        return config.get('spatial', res['spatial']), config.get(
            'freq', res['freq']
        )

    @classmethod
    def _get_meta(cls, config, run_type='main'):
        """Get meta file for a given extent, satellite, and resolution."""

        meta_file = config.get('meta_file', None)
        if meta_file is not None:
            return meta_file

        if 'final_spatial' in config:
            spatial = config['final_spatial']

        else:
            spatial = config.get('spatial', cls._get_res(config)[0])

        meta_file = f'nsrdb_meta_{spatial}'

        if config['year'] > 2017 and 'collect' not in run_type:
            msg = '"extent" key not provided. Provide "meta_file" or "extent"'
            assert 'extent' in config, msg
            meta_file += f'_{config["extent"]}'

        if run_type == 'main':
            msg = (
                '"satellite" key not provided. Provide "meta_file" or '
                '"satellite".'
            )
            assert 'satellite' in config, msg
            meta_file += f'_{config["satellite"]}_{config["lon_seam"]}'

        meta_file = os.path.join(
            config['meta_dir'], f'{os.path.basename(meta_file)}.csv'
        )

        return meta_file

    @classmethod
    def _get_config_file(cls, config, run_type='main'):
        """Get config file path for a given run type."""
        return os.path.join(
            config.get('out_dir', './'),
            f'config_{run_type.replace("-", "_")}.json',
        )

    @classmethod
    def _get_run_name(cls, config, run_type='main'):
        """Get name of run for given main run input."""

        run_name = config.get('run_name', None)
        if run_name is not None:
            return run_name

        config.update({
            k: v for k, v in BASE_KWARGS.items() if k not in config
        })
        pattern_dict = {
            'surfrad': cls.SURFRAD_RUN_NAME,
            'main': cls.MAIN_RUN_NAME,
            'blend': cls.BLEND_RUN_NAME,
            'aggregate': cls.AGG_RUN_NAME,
            'collect-aggregate': cls.COLLECT_AGG_RUN_NAME,
            'collect-blend': cls.COLLECT_BLEND_RUN_NAME,
        }
        pattern = pattern_dict[run_type]
        keys = get_format_keys(pattern)
        run_config = {k: v for k, v in config.items() if k in keys}
        if 'spatial' in keys or 'freq' in keys:
            run_config['spatial'], run_config['freq'] = cls._get_res(
                run_config
            )
        return pattern.format(**run_config)

    @classmethod
    def _update_run_templates(cls, config, run_type='main'):
        """Replace format keys and dictionary keys in config templates with
        user input values."""

        logger.info(
            'Updating NSRDB run templates with config:\n'
            f'{pprint.pformat(config, indent=2)}'
        )

        config['doy_range'] = config.get(
            'doy_range',
            ([1, 367] if calendar.isleap(config['year']) else [1, 366]),
        )
        config['start_doy'], config['end_doy'] = (
            config['doy_range'][0],
            config['doy_range'][1],
        )
        config['run_name'] = cls._get_run_name(config, run_type=run_type)
        config['out_dir'] = os.path.join(config['out_dir'], config['run_name'])

        template = (
            PRE2018_CONFIG_TEMPLATE
            if int(config['year']) < 2018
            else POST2017_CONFIG_TEMPLATE
        )
        with open(template, encoding='utf-8') as s:
            config_dict = json.loads(str_replace_dict(s.read(), config))

        config_dict.update({
            k: v for k, v in config.items() if k in config_dict
        })

        # special case for 2018. use all points as neighbors during cloud
        # regrid

        if config['year'] == 2018 and config['satellite'] == 'west':
            config_dict['data-model']['dist_lim'] = 1e6

        cls._write_config(config_dict, cls._get_config_file(config, 'nsrdb'))

        with open(PIPELINE_CONFIG_TEMPLATE, encoding='utf-8') as s:
            config_dict = json.loads(str_replace_dict(s.read(), config))

        cls._write_config(
            config_dict, cls._get_config_file(config, 'pipeline')
        )

        run_file = os.path.join(config['out_dir'], 'run.sh')
        with open(run_file, 'w') as f:
            f.write('python -m nsrdb.cli pipeline -c config_pipeline.json')

        logger.info(f'Saved run script: {run_file}.')

    @classmethod
    def surfrad(cls, kwargs):
        """Get basic config template specified parameters replaced."""
        config = cls.init_kwargs(kwargs, SURFRAD_KWARGS)
        config['extent_tag'] = EXTENT_MAP['extent_tag'][config['extent']]
        config['meta_file'] = os.path.join(
            config['meta_dir'], 'surfrad_meta.csv'
        )
        cls._update_run_templates(config, run_type='surfrad')

    @classmethod
    def main(cls, kwargs):
        """Modify config files with specified parameters

        Parameters
        ----------
        kwargs : dict
            Dictionary of parameters including year, basename, satellite,
            extent, freq, spatial, meta_file, doy_range
        """
        config = cls.init_kwargs(kwargs, MAIN_KWARGS)
        msg = (
            '"extent" key not provided. Provide "extent" so correct input '
            'data can be selected'
        )
        assert 'extent' in config, msg
        config['extent_tag'] = EXTENT_MAP['extent_tag'][config['extent']]
        config['lon_seam'] = EXTENT_MAP['lon_seam'][config['extent']]
        config['meta_file'] = cls._get_meta(config)
        config['spatial'], config['freq'] = cls._get_res(config)

        cls._update_run_templates(config)

    @classmethod
    def main_all(cls, kwargs):
        """Modify config files for all domains with specified parameters.

        Parameters
        ----------
        kwargs : dict
            Dictionary of parameters including year, basename, satellite,
            extent, freq, spatial, meta_file, doy_range
        """
        out_dir = os.path.abspath(kwargs.get('out_dir', './'))
        if kwargs['year'] < 2018:
            kwargs_list = [
                {'extent': 'full', 'satellite': sat}
                for sat in ('east', 'west')
            ]
        elif kwargs['year'] == 2018:
            kwargs_list = [
                {'extent': 'full', 'satellite': 'east', **NSRDB_2018['east']},
                {'extent': 'full', 'satellite': 'west', **NSRDB_2018['west']},
                {
                    'extent': 'conus',
                    'satellite': 'east',
                    **NSRDB_2018['conus'],
                },
            ]
        else:
            kwargs_list = [
                {'extent': ex, 'satellite': sat}
                for ex in ('full', 'conus')
                for sat in ('east', 'west')
            ]
        run_cmd = ''
        for kws in kwargs_list:
            input_kws = copy.deepcopy(kwargs)
            input_kws.update(kws)
            run_cmd += (
                f'cd {cls._get_run_name(input_kws)}; bash run.sh; cd ../; '
            )
            cls.main(input_kws)

        run_file = os.path.join(out_dir, 'run.sh')
        with open(run_file, 'w') as f:
            f.write(run_cmd)

        logger.info(f'Saved run script: {run_file}.')

    @classmethod
    def full(cls, kwargs):
        """Modify config files for all domains with specified parameters. Write
        all post processing config files and post processing pipeline config
        file for the given year

        Parameters
        ----------
        kwargs : dict
            Dictionary of parameters including year, basename, satellite,
            extent, freq, spatial, meta_file, doy_range
        """
        cls.main_all(kwargs)
        cls.post(kwargs)

    @classmethod
    def post(cls, kwargs):
        """Create all post processing config files for blending / aggregation /
        collection."""

        pipeline_config = {'pipeline': []}
        out_dir = os.path.abspath(kwargs.get('out_dir', './'))
        post_proc_dir = os.path.join(out_dir, 'post_proc')
        kwargs['post_proc_dir'] = kwargs.get('post_proc_dir', post_proc_dir)
        kwargs['data_dir'] = kwargs.get('data_dir', post_proc_dir)
        kwargs['main_dir'] = out_dir

        if kwargs['year'] > 2018:
            kwargs_list = [{'extent': 'conus'}, {'extent': 'full'}, None, None]
            step_names = [
                'blend-conus',
                'blend-full',
                'aggregate',
                'collect-aggregate',
            ]
        elif kwargs['year'] == 2018:
            kwargs_list = [None, None]
            step_names = ['aggregate', 'collect-aggregate']
        else:
            kwargs_list = [None, None]
            step_names = ['blend', 'collect-blend']

        for kws, step_name in zip(kwargs_list, step_names):
            fname = f'config_{step_name.replace("-", "_")}.json'
            mod_name = 'blend' if '_blend_' in fname else step_name
            func = getattr(cls, f'_{mod_name.replace("-", "_")}')
            config = func(kwargs if kws is None else {**kwargs, **kws})
            if mod_name == 'blend':
                config['out_dir'] = os.path.join(
                    post_proc_dir, config['run_name']
                )
            if mod_name == 'collect-blend':
                config['collect_dir'] = os.path.join(
                    post_proc_dir, config['run_name'].replace('_collect', '')
                )
            cls._write_config(
                config,
                os.path.join(post_proc_dir, fname),
                module_name=mod_name,
            )
            pipeline_config['pipeline'].append({
                step_name: fname,
                'command': mod_name,
            })
        cls._write_config(
            pipeline_config,
            cls._get_config_file({'out_dir': post_proc_dir}, 'pipeline_post'),
        )
        run_file = os.path.join(post_proc_dir, 'run.sh')
        with open(run_file, 'w') as f:
            f.write(
                'python -m nsrdb.cli pipeline -c config_pipeline_post.json'
            )

        logger.info(f'Saved run script: {run_file}.')

    @classmethod
    def _get_agg_entry(cls, config, extent):
        """Get entry in the aggregate data argument for a given extent."""
        meta_file = 'nsrdb_meta_{res}_{extent}.csv'
        tree_file = 'kdtree_nsrdb_meta_{res}_{extent}.pkl'

        if extent == 'final':
            source_priority = config.pop('source_priority')
            final_sub_dir = (
                f'{config.get("basename", "nsrdb")}_{config["final_spatial"]}'
                f'_{config["final_freq"]}'
            )
            return {
                'data_sub_dir': final_sub_dir,
                'fout': f'nsrdb_{config["year"]}.h5',
                'tree_file': tree_file.replace('_{extent}', '').format(
                    res=config['final_spatial']
                ),
                'meta_file': meta_file.replace('_{extent}', '').format(
                    res=config['final_spatial']
                ),
                'spatial': f'{config["final_spatial"]}',
                'freq': f'{config["final_freq"]}',
                'source_priority': source_priority,
            }

        return {
            'data_sub_dir': cls._get_run_name(
                {**config, 'extent': extent}, run_type='blend'
            ),
            'tree_file': tree_file.format(
                res=config[f'{extent}_spatial'], extent=extent
            ),
            'meta_file': meta_file.format(
                res=config[f'{extent}_spatial'], extent=extent
            ),
            'spatial': config[f'{extent}_spatial'],
            'freq': config[f'{extent}_freq'],
        }

    @classmethod
    def _aggregate(cls, kwargs):
        """Get config for conus and full disc high-resolution to low-resolution
        aggregation.  This is then used as the input to `nsrdb.cli.aggregate`

        Parameters
        ----------
        kwargs : dict
            Dictionary with keys specifying the case for which to aggregate
            files
        """
        config = cls.init_kwargs(kwargs, AGG_KWARGS)

        if config['year'] == 2018:
            data = NSRDB_2018

        else:
            data = {
                'full_disc': cls._get_agg_entry(config, extent='full'),
                'conus': cls._get_agg_entry(config, extent='conus'),
                'final': cls._get_agg_entry(config, extent='final'),
            }

        config['data'] = data
        config['run_name'] = cls._get_run_name(config, run_type='aggregate')
        return config

    @classmethod
    def aggregate(cls, kwargs):
        """Get config for conus and full disc high-resolution to low-resolution
        aggregation.  This is then used as the input to `nsrdb.cli.aggregate`

        Parameters
        ----------
        kwargs : dict
            Dictionary with keys specifying the case for which to aggregate
            files
        """

        config = cls._aggregate(kwargs)
        cls._write_config(
            config,
            os.path.join(
                kwargs.get('out_dir', './'),
                'config_aggregate.json',
            ),
            module_name='aggregate',
        )

    @classmethod
    def _blend(cls, kwargs):
        """Get config dictionary for nsrdb.cli.blend for standard NSRDB runs
        for a given extent.

        Parameters
        ----------
        kwargs : dict
            Dictionary with keys specifying the case for which to blend data
            files
        """
        config = cls.init_kwargs(kwargs, BLEND_KWARGS)
        config['map_col'] = EXTENT_MAP['map_col'][config['extent']]
        config['lon_seam'] = EXTENT_MAP['lon_seam'][config['extent']]
        config['meta_file'] = cls._get_meta(config, run_type='blend')

        config['east_dir'] = os.path.join(
            config['main_dir'],
            cls._get_run_name({'satellite': 'east', **config}),
            'final',
        )
        config['west_dir'] = os.path.join(
            config['main_dir'],
            cls._get_run_name({'satellite': 'west', **config}),
            'final',
        )
        config['run_name'] = cls._get_run_name(config, run_type='blend')

        return config

    @classmethod
    def _write_config(cls, config, config_file, module_name=None):
        """Write config to .json file."""

        config_dir = os.path.dirname(config_file)
        config_dir = config_dir if config_dir else config.get('out_dir', './')
        for k, v in config.items():
            if '_dir' in k:
                config[k] = os.path.abspath(v)

        os.makedirs(config_dir, exist_ok=True)

        if module_name is not None:
            exec_kwargs = config.pop('execution_control')
            config = {
                module_name: config,
                'execution_control': exec_kwargs,
            }
        with open(config_file, 'w') as f:
            f.write(json.dumps(config, indent=2))

        logger.info(
            f'Created config file {config_file}:'
            f'\n{pprint.pformat(config, indent=2)}'
        )

    @classmethod
    def blend(cls, kwargs):
        """Get config dictionary for nsrdb.cli.blend for standard NSRDB runs.

        Parameters
        ----------
        kwargs : dict
            Dictionary with keys specifying the case for which to blend data
            files
        """
        config = cls._blend(kwargs)
        cls._write_config(
            config,
            cls._get_config_file(kwargs, f'blend_{config["extent"]}'),
            module_name='blend',
        )

    @classmethod
    def _collect_blend(cls, kwargs):
        """Create config for collecting blended files into a single output
        file.

        Note
        ----
        This is used to combine year < 2018 dset files into single files with
        all dsets. e.g. clearsky, irradiance, clouds, etc -> nsrdb_{year}.h5

        Parameters
        ----------
        kwargs : dict
            Dictionary with keys specifying the case for blend collection
        """

        config = cls.init_kwargs(kwargs, COLLECT_BLEND_KWARGS)
        config['meta_final'] = cls._get_meta(config, run_type='collect-blend')
        config['collect_dir'] = os.path.join(
            config['out_dir'],
            'post_proc',
            cls._get_run_name(config, run_type='blend'),
        )
        config['fout'] = os.path.join(
            f'{config["out_dir"]}',
            f'{config["basename"]}_{config["year"]}.h5',
        )
        config['run_name'] = cls._get_run_name(
            config, run_type='collect-blend'
        )
        config['collect_tag'] = f'{config["basename"]}_{config["extent"]}'
        return config

    @classmethod
    def collect_blend(cls, kwargs):
        """Collect blended files into a single output file.

        Note
        ----
        This is used to combine year < 2018 dset files into single files with
        all dsets. e.g. clearsky, irradiance, clouds, etc -> nsrdb_{year}.h5

        Parameters
        ----------
        kwargs : dict
            Dictionary with keys specifying the case for blend collection
        """

        config = cls._collect_blend(kwargs)
        cls._write_config(
            config,
            cls._get_config_file(kwargs, 'collect_blend'),
            module_name='collect-blend',
        )

    @classmethod
    def _collect_aggregate(cls, kwargs):
        """Create config for aggregation collection

        Note
        ----
        This is used to collect single dset aggregated files into a single file
        with all dsets. e.g. e.g. clearsky, irradiance, clouds, etc ->
        nsrdb_{year}.h5

        Parameters
        ----------
        kwargs : dict
            Dictionary with keys specifying the case for aggregation collection
        """
        config = cls.init_kwargs(kwargs, COLLECT_AGG_KWARGS)

        config['meta_final'] = cls._get_meta(
            config, run_type='collect-aggregate'
        )
        config['collect_dir'] = os.path.join(
            config['post_proc_dir'],
            f'nsrdb_{config["final_spatial"]}_{config["final_freq"]}',
        )
        config['collect_tag'] = f'{config["basename"]}_'
        config['fout'] = os.path.join(
            f'{config["out_dir"]}',
            f'{config["basename"]}_{config["year"]}.h5',
        )
        config['run_name'] = cls._get_run_name(
            config, run_type='collect-aggregate'
        )

        return config

    @classmethod
    def collect_aggregate(cls, kwargs):
        """Create config for aggregation collection

        Note
        ----
        This is used to collect single dset aggregated files into a single file
        with all dsets. e.g. e.g. clearsky, irradiance, clouds, etc ->
        nsrdb_{year}.h5

        Parameters
        ----------
        kwargs : dict
            Dictionary with keys specifying the case for aggregation collection
        """
        config = cls._collect_aggregate(kwargs)
        cls._write_config(
            config,
            cls._get_config_file(kwargs, 'collect_aggregate'),
            module_name='collect-aggregate',
        )
