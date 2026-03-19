"""NSRDB Command Line Interface (CLI)."""

import itertools
import json
import logging
import os

import click
from gaps import Pipeline
from gaps.batch import BatchJob
from gaps.cli.pipeline import pipeline as gaps_pipeline
from rex import safe_json_load
from rex.utilities.loggers import init_logger

from nsrdb import __version__
from nsrdb.aggregation.aggregation import Manager
from nsrdb.blend.blend import Blender
from nsrdb.config.create_configs import CreateConfigs
from nsrdb.file_handlers.collection import Collector
from nsrdb.nsrdb import NSRDB
from nsrdb.tmy import TmyRunner
from nsrdb.utilities import ModuleName
from nsrdb.utilities.cli import BaseCLI
from nsrdb.utilities.file_utils import ts_freq_check

logger = logging.getLogger(__name__)


class DictOrFile(click.ParamType):
    """Dict or file click input argument type."""

    name = 'dict_or_file'

    @staticmethod
    def convert(value, param, ctx):
        """Convert to dict or return as None."""
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and os.path.exists(value):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception as e:
                msg = (
                    f'Could not load {value} as a dictionary. Make sure '
                    'you provide a valid string representation'
                )
                raise ValueError(msg) from e
        if value is None:
            return None
        raise TypeError(
            f'Cannot recognize input type: {value} {type(value)} {param} {ctx}'
        )


CONFIG_TYPE = DictOrFile()


@click.group()
@click.version_option(version=__version__)
@click.option(
    '--config',
    '-c',
    required=False,
    type=CONFIG_TYPE,
    help='NSRDB config file json or dict for a single module.',
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Flag to turn on debug logging. Default is False.',
)
@click.pass_context
def main(ctx, config, verbose):
    """NSRDB command line interface.

    Try using the following commands to pull up the help pages for the
    respective NSRDB CLIs::

        $ python -m nsrdb.cli --help

        $ python -m nsrdb.cli create-configs --help

        $ python -m nsrdb.cli pipeline --help

        $ python -m nsrdb.cli data-model --help

        $ python -m nsrdb.cli ml-cloud-fill --help

        $ python -m nsrdb.cli daily-all-sky --help

        $ python -m nsrdb.cli collect-data-model --help

        $ python -m nsrdb.cli tmy --help

        $ python -m nsrdb.cli blend --help

        $ python -m nsrdb.cli aggregate --help

    Each of these commands can be run with a config_file or a dictionary
    (represented as a string) provided through the `-c` argument. A typical
    config file might look like::

        \b
        {
            "logging": {"log_level": "DEBUG"},
            "<command name>": {'run_name': ...,
                               **kwargs},
            "direct": {more kwargs},
            "execution_control": {"option": "kestrel", ...}
            "another command": {...},
            ...
            ]
        }

    The "run_name" key will be prepended to each kicked off job. e.g.
    <run_name>_0, <run_name>_1, ... for multiple jobs from the same cli module.
    The "direct" key is used to provide arguments to multiple commands. This
    removes the need for duplication in the case of multiple commands having
    the same argument values. "execution_control" is used to provide arguments
    to the SLURM manager for HPC submissions or to select local execution with
    {"option": "local"}

    To do a standard CONUS / Full Disc run use the following commands::

        $ CONFIG='{"year": <year>, "out_dir": <out_dir>}'

        $ python -m nsrdb.cli create-configs -c ${CONFIG} -r full

        $ cd <out_dir>

        $ bash run.sh (run this until all main steps are complete)

        $ cd post_proc

        $ bash run.sh (run this until all post-proc steps are complete)

    See the help pages of the module CLIs for more details on the config files
    for each CLI.
    """  # noqa: D301
    ctx.ensure_object(dict)
    ctx.obj['CONFIG'] = config
    ctx.obj['VERBOSE'] = verbose
    ctx.obj['LOG_LEVEL'] = 'DEBUG' if verbose else 'INFO'

    init_logger('nsrdb', log_level=ctx.obj['LOG_LEVEL'], log_file=None)
    init_logger('gaps', log_level=ctx.obj['LOG_LEVEL'], log_file=None)


@main.group(invoke_without_command=True)
@click.option(
    '--config',
    '-c',
    required=True,
    type=click.Path(exists=True),
    help='NSRDB pipeline configuration json file.',
)
@click.option(
    '--cancel',
    is_flag=True,
    help='Flag to cancel all jobs associated with a given pipeline.',
)
@click.option(
    '--monitor',
    is_flag=True,
    help='Flag to monitor pipeline jobs continuously. '
    'Default is not to monitor (kick off jobs and exit).',
)
@click.option(
    '--background',
    is_flag=True,
    help='Flag to monitor pipeline jobs continuously in the '
    'background using the nohup command. This only works with the '
    '--monitor flag. Note that the stdout/stderr will not be '
    'captured, but you can set a pipeline log_file to capture logs.',
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Flag to turn on debug logging. Default is False.',
)
@click.pass_context
def pipeline(ctx, config, cancel, monitor, background, verbose):
    """Execute multiple steps in an NSRDB pipeline.

    Typically, a good place to start is to set up an nsrdb job with a pipeline
    config that points to several NSRDB modules that you want to run in serial.
    You would call the nsrdb pipeline CLI using::

        $ python -m nsrdb.cli -c config_pipeline.json pipeline

    A typical nsrdb pipeline config.json file might look like this::

        \b
        {
            "logging": {"log_level": "DEBUG"},
            "pipeline": [
                {"data-model": "./config_nsrdb.json"},
                {"ml-cloud-fill": "./config_nsrdb.json"},
                {"daily-all-sky": "./config_nsrdb.json"},
                {"collect-data-model": "./config_nsrdb.json"},
            ]
        }

    See the other CLI help pages for what the respective module configs
    require.
    """  # noqa: D301

    ctx.ensure_object(dict)
    ctx.obj['VERBOSE'] = verbose or ctx.obj.get('VERBOSE', False)
    gaps_pipeline(config, cancel, monitor, background)


@main.command()
@click.option(
    '--config',
    '-c',
    required=True,
    type=CONFIG_TYPE,
    help="""Either a path to a .json config file or a dictionary. Needs to
    include at least a "year" key. If input is a dictionary the dictionary
    needs to provided in a string format::

        $ '{"year": 2019, "freq": "5min"}'

    \b
    Available keys:
        year (year to run),
        freq (target time step. e.g. "5min"),
        out_dir (parent directory for run directory),
        satellite (east/west),
        spatial (meta file resolution, e.g. "2km" or "4km"),
        extent (full/conus),
        basename (string to prepend to files and job names),
        meta_file (e.g. "surfrad_meta.csv". Auto populated if None.),
        doy_range (All days of year if None).

    \b
    default_kwargs = {
        "basename": "nsrdb",
        "freq": "5min",
        "satellite": "east",
        "extent": "conus",
        "out_dir": "./",
        "spatial": "4km",
        "meta_file" : None,
        "doy_range": None
    }""",
)
@click.option(
    '--run_type',
    '-r',
    default='surfrad',
    type=str,
    help="""Run type to create configs for. Can be "surfrad" (just writes a
    single template config with any provided kwargs replaced, with a surfrad
    meta file), "full" (generates all config and pipline files for the given
    year, including all domain main runs, blending, aggregation, and
    collection), or "main" (for standard run without post-processing, with
    data-model, ml-cloud-fill, all-sky, and collect-data-model), "aggregate"
    (for aggregating post-2018 data to pre-2018 resolution), "blend" (for
    blending east and west domains into a single domain), or "post" (for all
    blending / aggregation / collection for a given year)""",
)
@click.option(
    '--all_domains',
    '-ad',
    is_flag=True,
    help="""Flag to generate config files for all domains. If True config files
    for east/west and conus/full will be generated. (just full if year is
    < 2018). satellite, extent, spatial, freq, and meta_file will be auto
    populated.""",
)
@click.option(
    '--collect',
    '-col',
    is_flag=True,
    help="""Flag to generate config files for module collection. This applies
    to run_type = "aggregate" or "blend".""",
)
@click.pass_context
def create_configs(
    ctx, config, run_type='surfrad', all_domains=False, collect=False
):
    """Create config files for standard NSRDB runs using config templates.

    To generate all full_disc / conus run directories for east /
    west regions, each with main routine config files contained run the
    following::

    $ CONFIG='{"year": 2020, "out_dir": "./"}'

    $ python -m nsrdb.cli create-configs --run_type full -c ${CONFIG}

    Additionally, conus / full_disc blend configs, aggregation config,
    collection config, and a post processing pipeline config with all these
    steps will be written to a "post_proc" directory so that post-processing
    can be run simply with::

    $ python -m nsrdb.cli pipeline -c config_pipeline_post.json
    """

    ctx.ensure_object(dict)
    func_name = f'collect_{run_type}' if collect else run_type
    func_name = 'main_all' if run_type == 'main' and all_domains else func_name
    valid_types = ['full', 'main', 'aggregate', 'blend', 'post', 'surfrad']
    msg = (
        f'Received unknown "run_type" {run_type}. Accepted values are '
        f'{valid_types}'
    )
    assert hasattr(CreateConfigs, func_name), msg
    config_func = getattr(CreateConfigs, func_name)
    config_func(config)


@main.command()
@click.option(
    '--config',
    '-c',
    type=CONFIG_TYPE,
    required=True,
    help='Path to config file or dict of kwargs for NSRDB.run_data_model()',
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Flag to turn on debug logging. Default is False.',
)
@click.pass_context
def data_model(ctx, config, verbose=False, pipeline_step=None):
    """Run daily data-model and save output files.

    You would call the nsrdb data-model module using::

        $ python -m nsrdb.cli -c config.json data-model

    A typical config.json file might look like this::

        \b
        {
            "collect-data-model": {...},
            "daily-all-sky": {...},
            "data-model": {
                "dist_lim": 2.0,
                "doy_range": [1, 367],
                "factory_kwargs": {
                  "cld_opd_dcomp": ...
                  "cld_press_acha": ...
                  "cld_reff_dcomp": ...
                  "cloud_fraction": ...
                  "cloud_probability": ...
                  "cloud_type": ...
                  "refl_0_65um_nom": ...
                  "refl_0_65um_nom_stddev_3x3": ...
                  "refl_3_75um_nom": ...
                  "surface_albedo": ...
                  "temp_11_0um_nom": ...
                  "temp_11_0um_nom_stddev_3x3": ...
                  "temp_3_75um_nom": ...
                },
                "max_workers": null,
                "max_workers_regrid": 16,
                "mlclouds": true
            },
            "direct": {
                "log_level": "INFO",
                "name": ...
                "freq": "5min"
                "grid": "/projects/pxs/reference_grids/surfrad_meta.csv,
                "out_dir": "./",
                "max_workers": 32,
                "year": "2018"
            },
            "execution_control": {
                "option": "kestrel",
                "alloc": "pxs",
                "feature": "--qos=normal",
                "walltime": 40
            },
            "ml-cloud-fill": {...}
        }

    See the other CLI help pages for what the respective module configs
    require.
    """  # noqa: D301

    config_dict = safe_json_load(config) if isinstance(config, str) else config
    direct = config_dict.get('direct', {})
    msg = 'Config must include "freq" key.'
    assert 'freq' in config_dict or 'freq' in direct, msg
    nsrdb_freq = config_dict.get('freq', direct['freq'])
    ts_freq_check(nsrdb_freq)

    BaseCLI.kickoff_multiday(
        ctx=ctx,
        module_name=ModuleName.DATA_MODEL,
        func=NSRDB.run_data_model,
        config=config,
        verbose=verbose,
        pipeline_step=pipeline_step,
    )


@main.command()
@click.option(
    '--config',
    '-c',
    type=CONFIG_TYPE,
    required=True,
    help='Path to config file or dict with kwargs for NSRDB.ml_cloud_fill()',
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Flag to turn on debug logging. Default is False.',
)
@click.pass_context
def ml_cloud_fill(ctx, config, verbose=False, pipeline_step=None):
    """Gap fill cloud properties using mlclouds.

    You would call the nsrdb ml-cloud-fill module using::

        $ python -m nsrdb.cli -c config.json ml-cloud-fill

    A typical config.json file might look like this::

        \b
        {
            "collect-data-model": {...},
            "daily-all-sky": {...},
            "data-model": {...},
            "direct": {...},
            "execution_control": {
                "option": "kestrel",
                "alloc": "pxs",
                "feature": "--qos=normal",
                "walltime": 40
            },
            "ml-cloud-fill": {
                "col_chunk": 10000,
                "fill_all": false,
                "max_workers": 4
                "model_path": ...
            }
        }

    See the other CLI help pages for what the respective module configs
    require.
    """  # noqa: D301
    BaseCLI.kickoff_multiday(
        ctx=ctx,
        module_name=ModuleName.ML_CLOUD_FILL,
        func=NSRDB.ml_cloud_fill,
        config=config,
        verbose=verbose,
        pipeline_step=pipeline_step,
    )


@main.command()
@click.option(
    '--config',
    '-c',
    type=CONFIG_TYPE,
    required=True,
    help=(
        'Path to config file or dict with kwargs for '
        'NSRDB.run_daily_all_sky()'
    ),
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Flag to turn on debug logging. Default is False.',
)
@click.pass_context
def daily_all_sky(ctx, config, verbose=False, pipeline_step=None):
    """Run all-sky physics model on daily data-model output.

    You would call the nsrdb daily-all-sky module using::

        $ python -m nsrdb.cli -c config.json daily-all-sky

    A typical config.json file might look like this::

        \b
        {
            "collect-data-model": {...},
            "daily-all-sky": {
                "disc_on": false,
                "out_dir": "./all_sky",
                "year": 2018,
                "grid": "/projects/pxs/reference_grids/surfrad_meta.csv",
                "freq": "5min"
            },
            "data-model": {...},
            "direct": {...},
            "execution_control": {
                "option": "kestrel",
                "alloc": "pxs",
                "feature": "--qos=normal",
                "walltime": 40
            },
            "ml-cloud-fill": {...}
        }
    """  # noqa : D301
    BaseCLI.kickoff_multiday(
        ctx=ctx,
        module_name=ModuleName.DAILY_ALL_SKY,
        func=NSRDB.run_daily_all_sky,
        config=config,
        verbose=verbose,
        pipeline_step=pipeline_step,
    )


@main.command()
@click.option(
    '--config',
    '-c',
    type=CONFIG_TYPE,
    required=True,
    help='Path to config file or dict with kwargs for NSRDB.cloud_fill()',
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Flag to turn on debug logging. Default is False.',
)
@click.pass_context
def cloud_fill(ctx, config, verbose=False, pipeline_step=None):
    """Gap fill cloud properties in a collect-data-model output file, using
    legacy gap-fill method."""

    BaseCLI.kickoff_multichunk(
        ctx=ctx,
        module_name=ModuleName.CLOUD_FILL,
        func=NSRDB.gap_fill_clouds,
        config=config,
        verbose=verbose,
        pipeline_step=pipeline_step,
    )


@main.command()
@click.option(
    '--config',
    '-c',
    type=CONFIG_TYPE,
    required=True,
    help='Path to config file or dict with kwargs for NSRDB.run_all_sky()',
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Flag to turn on debug logging. Default is False.',
)
@click.pass_context
def all_sky(ctx, config, verbose=False, pipeline_step=None):
    """Run all-sky physics model on collected data model output files."""

    BaseCLI.kickoff_multichunk(
        ctx=ctx,
        module_name=ModuleName.ALL_SKY,
        func=NSRDB.run_all_sky,
        config=config,
        verbose=verbose,
        pipeline_step=pipeline_step,
    )


@main.command()
@click.option(
    '--config',
    '-c',
    type=CONFIG_TYPE,
    required=True,
    help=(
        'Path to config file or dict with kwargs for '
        'NSRDB.collect_data_model()'
    ),
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Flag to turn on debug logging. Default is False.',
)
@click.pass_context
def collect_data_model(ctx, config, verbose=False, pipeline_step=None):
    """Collect data-model output files to a single site-chunked output file.

    You would call the nsrdb collect-data-model module using::

        $ python -m nsrdb.cli -c config.json collect-data-model

    A typical config.json file might look like this::

        \b
        {
            "collect-data-model": {
                "final": true,
                "max_workers": 10,
                "n_chunks": 1,
                "memory": 178,
                "n_writes": 1,
                "walltime": 48
            },
            "daily-all-sky": {...},
            "data-model": {...},
            "direct": {...},
            "execution_control": {
                "option": "kestrel",
                "alloc": "pxs",
                "feature": "--qos=normal",
                "walltime": 40
            },
            "ml-cloud-fill": {...}
        }
    """  # noqa : D301
    config = BaseCLI.from_config_preflight(
        ctx=ctx,
        module_name=ModuleName.COLLECT_DATA_MODEL,
        config=config,
        verbose=verbose,
        pipeline_step=pipeline_step,
    )

    config['n_chunks'] = config.get('n_chunks', 1)
    config['n_writes'] = config.get('n_writes', 1)
    config['final'] = config.get('final', False)
    fnames = sorted(NSRDB.OUTS.keys())
    min_files = [
        f for f in fnames if f.split('_')[1] not in ('clearsky', 'irradiance')
    ]
    collect_files = fnames if config['final'] else min_files

    msg = 'collect-data-model was marked as final but n_chunks != 1'
    assert not (config['final'] and config['n_chunks'] != 1), msg

    for i_chunk, fname in itertools.product(
        range(config['n_chunks']), collect_files
    ):
        log_id = '_'.join(fname.split('_')[1:-1] + [str(i_chunk)])
        config['i_chunk'] = i_chunk
        config['i_fname'] = fnames.index(fname)
        config['job_name'] = f'{ctx.obj["RUN_NAME"]}_{log_id}'

        BaseCLI.kickoff_job(
            ctx=ctx,
            module_name=ModuleName.COLLECT_DATA_MODEL,
            func=NSRDB.collect_data_model,
            config=config,
            log_id=log_id,
        )


@main.command()
@click.option(
    '--config',
    '-c',
    type=CONFIG_TYPE,
    required=True,
    help='Path to config file or dict with kwargs for NSRDB.collect_final()',
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Flag to turn on debug logging. Default is False.',
)
@click.pass_context
def collect_final(ctx, config, verbose=False, pipeline_step=None):
    """Collect chunked files with final data into final full files."""

    config = BaseCLI.from_config_preflight(
        ctx=ctx,
        module_name=ModuleName.COLLECT_FINAL,
        config=config,
        verbose=verbose,
        pipeline_step=pipeline_step,
    )

    for i_fname, fname in enumerate(sorted(NSRDB.OUTS.keys())):
        log_id = '_'.join(fname.split('_')[1:-1])
        config['job_name'] = f'{ctx.obj["RUN_NAME"]}_{log_id}'
        config['i_fname'] = i_fname
        BaseCLI.kickoff_job(
            ctx=ctx,
            module_name=ModuleName.COLLECT_FINAL,
            func=NSRDB.collect_final,
            config=config,
            log_id=log_id,
        )


@main.command()
@click.option(
    '--config',
    '-c',
    type=CONFIG_TYPE,
    required=True,
    help='Path to config file or dict with kwargs for NSRDB.collect_daily()',
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Flag to turn on debug logging. Default is False.',
)
@click.pass_context
def collect_daily(ctx, config, verbose=False, pipeline_step=None):
    """Collect daily files into a final file."""

    BaseCLI.kickoff_single(
        ctx=ctx,
        module_name=ModuleName.COLLECT_DAILY,
        func=Collector.collect_daily,
        config=config,
        verbose=verbose,
        pipeline_step=pipeline_step,
    )


@main.command()
@click.option(
    '--config',
    '-c',
    type=str,
    required=True,
    help='Path to config file with kwargs for NSRDB.blend_files()',
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Flag to turn on debug logging. Default is False.',
)
@click.pass_context
def blend(ctx, config, verbose=False, pipeline_step=None):
    """Blend files from separate domains (e.g. east / west) into a single
    domain."""

    config = BaseCLI.from_config_preflight(
        ctx=ctx,
        module_name=ModuleName.BLEND,
        config=config,
        verbose=verbose,
        pipeline_step=pipeline_step,
    )

    file_tags = config.get('file_tag', 'all')
    file_tags = (
        file_tags
        if file_tags != 'all'
        else ['_'.join(k.split('_')[1:-1]) for k in NSRDB.OUTS]
    )

    file_tags = file_tags if isinstance(file_tags, list) else [file_tags]
    for file_tag in file_tags:
        log_id = file_tag
        config['job_name'] = f'{ctx.obj["RUN_NAME"]}_{log_id}'
        config['file_tag'] = file_tag
        BaseCLI.kickoff_job(
            ctx=ctx,
            module_name=ModuleName.BLEND,
            func=Blender.run_full,
            config=config,
            log_id=log_id,
        )


@main.command()
@click.option(
    '--config',
    '-c',
    type=str,
    required=True,
    help='Path to config file with kwargs for Collector.collect_dir()',
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Flag to turn on debug logging. Default is False.',
)
@click.pass_context
def collect_blend(ctx, config, verbose=False, pipeline_step=None):
    """Collect blended files into a single file with multiple datasets."""

    BaseCLI.kickoff_single(
        ctx=ctx,
        module_name=ModuleName.COLLECT_BLEND,
        func=Collector.collect_dir,
        config=config,
        verbose=verbose,
        pipeline_step=pipeline_step,
    )


@main.command()
@click.option(
    '--config',
    '-c',
    type=str,
    required=True,
    help='Path to config file with kwargs for NSRDB.aggregate_files()',
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Flag to turn on debug logging. Default is False.',
)
@click.pass_context
def aggregate(ctx, config, verbose=False, pipeline_step=None):
    """Aggregate data files to a lower resolution.

    NOTE: Used to create data files from high-resolution years (2018+) which
    match resolution of low-resolution years (pre 2018)
    """

    BaseCLI.kickoff_multichunk(
        ctx=ctx,
        module_name=ModuleName.AGGREGATE,
        func=Manager.run_chunk,
        config=config,
        verbose=verbose,
        pipeline_step=pipeline_step,
    )


@main.command()
@click.option(
    '--config',
    '-c',
    type=str,
    required=True,
    help='Path to config file with kwargs for Collector.collect_dir()',
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Flag to turn on debug logging. Default is False.',
)
@click.pass_context
def collect_aggregate(ctx, config, verbose=False, pipeline_step=None):
    """Collect aggregate data files into a single file with multiple
    datasets."""

    BaseCLI.kickoff_single(
        ctx=ctx,
        module_name=ModuleName.COLLECT_AGGREGATE,
        func=Collector.collect_dir,
        config=config,
        verbose=verbose,
        pipeline_step=pipeline_step,
    )


def _run_or_collect_tmy(
    ctx, config, verbose=False, pipeline_step=None, collect=False
):
    mod_name = ModuleName.COLLECT_TMY if collect else ModuleName.TMY
    config = BaseCLI.from_config_preflight(
        ctx=ctx,
        module_name=mod_name,
        config=config,
        verbose=verbose,
        pipeline_step=pipeline_step,
    )
    tmy_types = config.pop('tmy_types', ['tmy', 'tgy', 'tdy'])
    fn_out = config.pop('fn_out', 'tmy.h5')
    out_dir = config['out_dir']
    for tmy_type in tmy_types:
        func = TmyRunner.collect if collect else TmyRunner.tmy
        config['tmy_type'] = tmy_type
        config['out_dir'] = os.path.join(out_dir, f'{tmy_type}/')
        config['job_name'] = f'{ctx.obj["RUN_NAME"]}_{tmy_type}'

        if '{}' in fn_out:
            config['fn_out'] = fn_out.format(tmy_type)
        else:
            config['fn_out'] = fn_out.replace('.h5', f'_{tmy_type}.h5')

        for node_index in range(config.get('n_nodes', 1)):
            log_id = f'{tmy_type}_{node_index}'
            config['job_name'] = f'{ctx.obj["RUN_NAME"]}_{log_id}'
            config['node_index'] = node_index
            BaseCLI.kickoff_job(
                ctx=ctx,
                module_name=mod_name,
                func=func,
                config=config,
                log_id=tmy_type,
            )


@main.command()
@click.option(
    '--config',
    '-c',
    type=str,
    required=True,
    help='Path to config file with kwargs for TmyRunner.tmy()',
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Flag to turn on debug logging. Default is False.',
)
@click.pass_context
def tmy(ctx, config, verbose=False, pipeline_step=None):
    """Create tmy files for given input files.

    You would call the nsrdb tmy module using::

        $ python -m nsrdb.cli -c config.json tmy

    A typical config.json file might look like this::

        \b
        {
            "tmy": {},
            "collect-tmy": {"purge_chunks": True},
            "direct": {
                "sites_per_worker": 50,
                "site_slice": [0, 100],
                "tmy_types": ['tmy', 'tdy', 'tgy'],
                "nsrdb_base_fp": './nsrdb_*_{}.h5',
                "years": [2000, ..., 2022],
                "out_dir": './",
                "fn_out": 'tmy_2000_2022.h5'
            }
        }
    """  # noqa : D301

    _run_or_collect_tmy(
        ctx,
        config,
        verbose=verbose,
        pipeline_step=pipeline_step,
        collect=False,
    )


@main.command()
@click.option(
    '--config',
    '-c',
    type=str,
    required=True,
    help='Path to config file with kwargs for TmyRunner.collect()',
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Flag to turn on debug logging. Default is False.',
)
@click.pass_context
def collect_tmy(ctx, config, verbose=False, pipeline_step=None):
    """Collect the previously generated tmy file chunks.

    You would call the nsrdb collect-tmy module using::

        $ python -m nsrdb.cli -c config.json collect-tmy

    A typical config.json file might look like this::

        \b
        {
            "tmy": {},
            "collect-tmy": {"purge_chunks": True},
            "direct": {
                "sites_per_worker": 50,
                "site_slice": [0, 100],
                "tmy_types": ['tmy', 'tdy', 'tgy'],
                "nsrdb_base_fp": './nsrdb_*_{}.h5',
                "years": [2000, ..., 2022],
                "out_dir": './",
                "fn_out": 'tmy_2000_2022.h5'
            }
        }
    """  # noqa : D301

    _run_or_collect_tmy(
        ctx, config, verbose=verbose, pipeline_step=pipeline_step, collect=True
    )


@main.group(invoke_without_command=True)
@click.option(
    '--config',
    '-c',
    required=True,
    type=click.Path(exists=True),
    help='NSRDB batch configuration json or csv file.',
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Flag to do a dry run (make batch dirs without running).',
)
@click.option(
    '--cancel',
    is_flag=True,
    help='Flag to cancel all jobs associated with a given pipeline.',
)
@click.option(
    '--delete',
    is_flag=True,
    help='Flag to delete all batch job sub directories associated '
    'with the batch_jobs.csv in the current batch config directory.',
)
@click.option(
    '--monitor-background',
    is_flag=True,
    help='Flag to monitor all batch pipelines continuously '
    'in the background using the nohup command. Note that the '
    'stdout/stderr will not be captured, but you can set a '
    'pipeline "log_file" to capture logs.',
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Flag to turn on debug logging. Default is False.',
)
@click.pass_context
def batch(
    ctx, config, dry_run, cancel, delete, monitor_background, verbose=False
):
    """Create and run multiple NSRDB project directories based on batch
    permutation logic.

    The NSRDB batch module (built on the gaps batch functionality) is a way to
    create and run many NSRDB pipeline projects based on permutations of
    key-value pairs in the run config files. A user configures the batch file
    by creating one or more "sets" that contain one or more arguments (keys
    found in config files) that are to be parameterized. For example, in the
    config below, two NSRDB pipelines will be created where year is set to
    2020 and 2021 in config_nsrdb.json::

        \b
        {
            "pipeline_config": "./config_pipeline.json",
            "sets": [
              {
                "args": {
                  "year": [2020, 2021],
                },
                "files": ["./config_nsrdb.json"],
                "set_tag": "set1"
              }
        }

    Run the batch module with::

        $ python -m nsrdb.cli -c config_batch.json batch

    Note that you can use multiple "sets" to isolate parameter permutations.
    """  # noqa : D301
    ctx.ensure_object(dict)
    ctx.obj['VERBOSE'] = verbose or ctx.obj.get('VERBOSE', False)
    batch = BatchJob(config)

    if cancel:
        batch.cancel()
    elif delete:
        batch.delete()
    else:
        batch.run(dry_run=dry_run, monitor_background=monitor_background)


Pipeline.COMMANDS[ModuleName.DATA_MODEL] = data_model
Pipeline.COMMANDS[ModuleName.CLOUD_FILL] = cloud_fill
Pipeline.COMMANDS[ModuleName.ALL_SKY] = all_sky
Pipeline.COMMANDS[ModuleName.DAILY_ALL_SKY] = daily_all_sky
Pipeline.COMMANDS[ModuleName.ML_CLOUD_FILL] = ml_cloud_fill
Pipeline.COMMANDS[ModuleName.BLEND] = blend
Pipeline.COMMANDS[ModuleName.AGGREGATE] = aggregate
Pipeline.COMMANDS[ModuleName.COLLECT_DATA_MODEL] = collect_data_model
Pipeline.COMMANDS[ModuleName.COLLECT_FINAL] = collect_final
Pipeline.COMMANDS[ModuleName.COLLECT_DAILY] = collect_daily
Pipeline.COMMANDS[ModuleName.TMY] = tmy
Pipeline.COMMANDS[ModuleName.COLLECT_BLEND] = collect_blend
Pipeline.COMMANDS[ModuleName.COLLECT_AGGREGATE] = collect_aggregate
Pipeline.COMMANDS[ModuleName.COLLECT_TMY] = collect_tmy


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.error('Error running NSRDB CLI.')
        raise
