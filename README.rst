.. start

##########################################################
Welcome to the National Solar Radiation Data Base (NSRDB)!
##########################################################
|Docs| |Tests| |Linter| |PyPi| |PythonV| |Codecov| |Zenodo|

.. |Docs| image:: https://github.com/NREL/nsrdb/workflows/Documentation/badge.svg
    :target: https://nrel.github.io/nsrdb/

.. |Tests| image:: https://github.com/NREL/nsrdb/workflows/Pytests/badge.svg
    :target: https://github.com/NREL/nsrdb/actions?query=workflow%3A%22Pytests%22

.. |Linter| image:: https://github.com/NREL/nsrdb/workflows/Lint%20Code%20Base/badge.svg
    :target: https://github.com/NREL/nsrdb/actions?query=workflow%3A%22Lint+Code+Base%22

.. |PyPi| image:: https://img.shields.io/pypi/pyversions/NREL-nsrdb.svg
    :target: https://pypi.org/project/NREL-nsrdb/

.. |PythonV| image:: https://badge.fury.io/py/NREL-nsrdb.svg
    :target: https://badge.fury.io/py/NREL-nsrdb

.. |Codecov| image:: https://codecov.io/gh/nrel/nsrdb/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/nrel/nsrdb

.. |Zenodo| image:: https://zenodo.org/badge/340209614.svg
    :target: https://zenodo.org/badge/latestdoi/340209614


The National Solar Radiation Database (NSRDB) software includes all the methods
for the irradiance data processing pipeline. To get started, check out the
NSRDB command line interface `(CLI)
<https://nrel.github.io/nsrdb/_cli/nsrdb.html#nsrdb>`_. Refer to the `NREL
website <https://nsrdb.nrel.gov/>`_ and the original `journal article
<https://www.sciencedirect.com/science/article/pii/S136403211830087X>`_ for
more information on the NSRDB.  For details on NSRDB variable units,
datatypes, interpolation methods, and other attributes, see the `NSRDB
variable meta data
<https://github.com/NREL/nsrdb/tree/main/nsrdb/config/nsrdb_vars.csv>`_ and
`NSRDB variable descriptions
<https://github.com/NREL/nsrdb/tree/main/nsrdb/config/var_descriptions.csv>`_.

The PXS All-Sky Irradiance Model
================================
The PXS All-Sky `Irradiance Model
<https://github.com/NREL/nsrdb/tree/main/nsrdb/all_sky>`_ is the main physics
package that calculates surface irradiance variables.

The NSRDB Data Model
====================
The NSRDB `Data Model
<https://github.com/NREL/nsrdb/tree/main/nsrdb/data_model>`_ is the data
aggregation framework that sources, processes, and prepares data for input to
All-Sky.

The MLClouds Model
==================
The `MLClouds Model <https://github.com/NREL/mlclouds.git>`_ is used to predict
missing cloud properties (a.k.a. Gap Fill). The NSRDB interface with MLClouds
can be found `here <https://github.com/NREL/nsrdb/tree/main/nsrdb/gap_fill>`_.

Installation
============

NOTE: The installation instructions below assume that you have python installed
on your machine and are using either `conda <https://docs.conda.io/en/latest/index.html>`__
or `pixi <https://pixi.sh/latest/>`_ as your package/environment manager.

Option 1: Install from PIP (recommended for analysts):
------------------------------------------------------

1. Create a new environment: ``conda create --name nsrdb python=3.11``

2. Activate environment: ``conda activate nsrdb``

3. Install nsrdb: ``pip install NREL-nsrdb``

Option 2: Clone repo (recommended for developers)
-------------------------------------------------

1. Run ``git clone git@github.com:NREL/nsrdb.git``
2. ``cd nsrdb``.
3. Make sure the branch is correct (install from main!)
4. If you are using conda, create and activate a new environment:
   ``conda create --name nsrdb python=3.11`` and ``conda activate nsrdb``

   4.1 Install ``nsrdb`` and its dependencies by running: ``pip install .`` (or ``pip install -e .`` for editable install)

5. Alternatively, run ``pixi install``
6. *Optional*: Set up the pre-commit hooks with ``pip install pre-commit`` or ``pixi add pre-commit`` and ``pre-commit install``

NSRDB Versions
==============

.. list-table::
    :widths: auto
    :header-rows: 1

    * - Version
      - Effective Date
      - Data Years*
      - Notes
    * - 4.1.1
      - 10/28/24
      - None
      - Integration with extended MLClouds models.
        Extended models can perform both cloud type and cloud property predictions.
    * - 4.1.0
      - 7/9/24
      - None
      - Complete CLI refactor.
    * - 4.0.0
      - 5/1/23
      - GOES 1998-2024, Meteosat 2005-2022.
      - Integrated an improved direct normal irradiance model (FARMS-DNI),
        described in the paper "Integration of a physics-based direct
        normal irradiance (DNI) model to enhance the National Solar
        Radiation Database (NSRDB)"
    * - 3.2.3
      - 4/13/23
      - None
      - Fixed MERRA interpolation issue #51 and deprecated python 3.7/3.8.
        Added changes to accommodate pandas v2.0.0.
    * - 3.2.2
      - 2/25/2022
      - 1998-2021
      - Implemented a model for snowy albedo as a function of temperature from
        MERRA2 based on the paper "A comparison of simulated and observed
        fluctuations in summertime Arctic surface albedo" by Becky Ross and
        John E. Walsh
    * - 3.2.1
      - 1/12/2021
      - 2021
      - Implemented an algorithm to re-map the parallax and shading corrected
        cloud coordinates to the nominal GOES coordinate system. This fixes the
        issue of PC cloud coordinates conflicting with clearsky coordinates.
        This also fixes the strange pattern that was found in the long term
        means generated from PC data.
    * - 3.2.0
      - 3/17/2021
      - 2020
      - Enabled cloud solar shading coordinate adjustment by default, enabled
        MLClouds machine learning gap fill method for missing cloud properties
        (cloud fill flag #7)
    * - 3.1.2
      - 6/8/2020
      - 2020
      - Added feature to adjust cloud coordinates based on solar position and
        shading geometry.
    * - 3.1.1
      - 12/5/2019
      - 2018+, TMY/TDY/TGY-2018
      - Complete refactor of TMY processing code.
    * - 3.1.0
      - 9/23/2019
      - 2018+
      - Complete refactor of NSRDB processing code for NSRDB 2018
    * - 3.0.6
      - 4/23/2019
      - 1998-2017
      - Missing data for all cloud properties gap filled using heuristics method
    * - 3.0.5
      - 4/8/2019
      - 1998-2017
      - Cloud pressure attributes and scale/offset fixed for 2016 and 2017
    * - 3.0.4
      - 3/29/2019
      - 1998-2017
      - Aerosol optical depth patched with physical range from 0 to 3.2
    * - 3.0.3
      - 2/25/2019
      - 1998-2017
      - Wind data recomputed to fix corrupted data in western extent
    * - 3.0.2
      - 2/25/2019
      - 1998-2017
      - Air temperature data recomputed from MERRA2 with elevation correction
    * - 3.0.1
      - 2018
      - 2017+
      - Moved from timeshift of radiation to timeshift of cloud properties.
    * - 3.0.0
      - 2018
      - 1998-2017
      - Initial release of PSM v3
        - Hourly AOD (1998-2016) from Modern-Era Retrospective analysis for
          Research and Applications Version 2 (MERRA2).
        - Snow-free Surface Albedo from MODIS (2001-2015), (MCD43GF CMG
          Gap-Filled Snow-Free Products from University of Massachusetts,
          Boston).
        - Snow cover from Integrated Multi-Sensor Snow and Ice Mapping System
          (IMS) daily snow cover product (National Snow and Ice Data Center).
        - GOES-East time-shift applied to cloud properties instead of solar
          radiation.
        - Modern-Era Retrospective analysis for Research and Applications,
          Version 2 (MERRA-2) is used for ancillary data (pressure, humidity,
          wind speed etc.)
    * - 2.0.0
      - 2016
      - 1998-2015
      - Initial release of PSM v2 (use of FARMS, downscaling of ancillary data
        introduced to account for elevation, NSRDB website distribution
        developed)
        - Clear sky: REST2, Cloudy sky: NREL FARMS model and DISC model
        - Climate Forecast System Reanalysis (CFSR) is used for ancillary data
        - Monthly 0.5º aerosol optical depth (AOD) for 1998-2014 using
          satellite and ground-based measurements. Monthly results interpolated
          to daily 4-km AOD data. Daily data calibrated using ground
          measurements to develop accurate AOD product.
    * - 1.0.0
      - 2015
      - 2005-2012
      - Initial release of PSM v1 (no FARMS)
        - Satellite Algorithm for Shortwave Radiation Budget (SASRAB) model
        - MMAC model for clear sky condition
        - The DNI for cloud scenes is then computed using the DISC model


Recommended Citation
====================
Update with current version and DOI:

Grant Buster, Brandon Benton, Mike Bannister, Yu Xie, Aron Habte, Galen
Maclaurin, Manajit Sengupta. National Solar Radiation Database (NSRDB).
https://github.com/NREL/nsrdb (version v4.0.0), 2023. DOI:
10.5281/zenodo.10471523

Acknowledgments
===============
This work (SWR-23-77) was authored by the National Renewable Energy Laboratory,
operated by Alliance for Sustainable Energy, LLC, for the U.S. Department of
Energy (DOE) under Contract No. DE-AC36-08GO28308. Funding provided by the DOE
Grid Deployment Office (GDO), the DOE Advanced Scientific Computing Research
(ASCR) program, the DOE Solar Energy Technologies Office (SETO), the DOE Wind
Energy Technologies Office (WETO), the United States Agency for International
Development (USAID), and the Laboratory Directed Research and Development
(LDRD) program at the National Renewable Energy Laboratory. The research was
performed using computational resources sponsored by the Department of Energy's
Office of Energy Efficiency and Renewable Energy and located at the National
Renewable Energy Laboratory. The views expressed in the article do not
necessarily represent the views of the DOE or the U.S. Government. The U.S.
Government retains and the publisher, by accepting the article for publication,
acknowledges that the U.S. Government retains a nonexclusive, paid-up,
irrevocable, worldwide license to publish or reproduce the published form of
this work, or allow others to do so, for U.S. Government purposes.

\*Note: The “Data Years” column shows which years of NSRDB data were updated at
the time of version release. However, each NSRDB file should be checked for the
version attribute, which should be a more accurate record of the actual data
version.
