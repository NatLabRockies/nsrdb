"""Download MERRA data for cloud, surface, or aerosol properties"""

import argparse
import calendar
import datetime as dt
import os

import requests
from bs4 import BeautifulSoup

cmd_pattern = 'wget --load-cookies ~/.urs_cookies --save-cookies '
cmd_pattern += '~/.urs_cookies --auth-no-challenge=on --keep-session-cookies '
cmd_pattern += '--content-disposition -O "{outfile}" "{line}"'


def get_var_list(source):
    """Get list of variables for given source. Source can be either rad, slv,
    or aer"""

    if source.lower() == 'rad':
        VARIABLES = [
            'CLDHGH',
            'CLDLOW',
            'CLDMID',
            'CLDTOT',
            'TAUHIGH',
            'TAULOW',
            'TAUMID',
            'TAUTOT',
        ]

    elif source.lower() == 'slv':
        VARIABLES = [
            'PS',
            'QV2M',
            'T2M',
            'TO3',
            'TQV',
            'U2M',
            'V2M',
            'U10M',
            'V10M',
        ]

    elif source.lower() == 'aer':
        VARIABLES = ['TOTANGSTR', 'TOTEXTTAU', 'TOTSCATAU']

    else:
        msg = (
            f'Unrecognized merra source type: {source}. '
            'This must be either "rad", "slv", or "aer"'
        )
        raise ValueError(msg)
    VARIABLES = '%2C'.join(VARIABLES)
    return VARIABLES


def get_url_pattern(source):
    """Get url pattern for given source. Source can be either rad, slv, or
    aer"""
    url_pattern = (
        'https://goldsmr4.gesdisc.eosdis.nasa.gov/daac-bin/OTF/'
        'HTTP_services.cgi?FILENAME=%2Fdata%2FMERRA2%2F'
        + 'M2T1NX'
        + source.upper()
        + '.5.12.4'
        '%2F{year}%2F{month}%2F{merra_prefix}'
        + '.tavg1_2d_'
        + source.lower()
        + '_Nx.'
        '{year}{month}{day}.nc4&FORMAT=bmM0Lw&SERVICE=L34RS_MERRA2'
        '&SHORTNAME=M2T1NXSLV&VERSION=1.02&VARIABLES='
        + get_var_list(source)
        + '&LABEL={merra_prefix}.tavg1_2d_'
        + source.lower()
        + '_Nx.{year}{month}{day}.SUB.nc'
        '&BBOX=-90%2C-180%2C90%2C180&DATASET_VERSION=5.12.4'
    )
    return url_pattern


def get_merra_file_list(year, month, source, ext='nc4'):
    """Get the list of merra files available"""
    url = (
        'https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/'
        f'M2T1NX{source.upper()}.5.12.4/{year}/{month}/'
    )
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    return [
        url + '/' + node.get('href')
        for node in soup.find_all('a')
        if node.get('href').endswith(ext)
    ]


def get_date(year, day):
    """Get string date format for given year and day"""
    date = dt.datetime(year, 1, 1) + dt.timedelta(day - 1)
    date = date.strftime('%Y%m%d')
    return date


def download_merra(date, source):
    """Download merra file for the given date string"""
    year = date[:4]
    month = date[4:6]
    day = date[6:8]

    files = get_merra_file_list(year=year, month=month, source=source)
    merra_prefix = files[0]
    merra_prefix = merra_prefix.split('/')[-1].split('.')[0]

    url_pattern = get_url_pattern(source=source)
    line = url_pattern.format(
        year=year, month=month, day=day, merra_prefix=merra_prefix
    )
    out_dir = f'/projects/pxs/ancillary/merra/tavg1_2d_{source.lower()}_Nx/'

    if 'LABEL=' in line:
        outfile = line.split('LABEL=')
        outfile = outfile[1].split('&')[0]
        line = line.strip('\n')
        outfile = f'{out_dir}/{outfile}'
        if not os.path.exists(outfile):
            print('Downloading {outfile}.')
            new_cmd = cmd_pattern.format(line=line, outfile=outfile)
            os.system(new_cmd)
        else:
            print(f'{outfile} already exists')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download MERRA-2 data for a given year and source type.'
    )
    parser.add_argument(
        '--year', type=int, required=True, help='Year to download.'
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Source type to download. This is either "rad", "slv", or "aer"',
    )
    args = parser.parse_args()

    days = range(1, 367) if calendar.isleap(args.year) else range(1, 366)

    for day in days:
        date = get_date(args.year, day)
        download_merra(date, args.source)
