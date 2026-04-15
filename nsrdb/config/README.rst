NSRDB Dataset Config / Meta Data
---------------------------------
This `config directory <https://github.com/NatLabRockies/nsrdb/tree/main/nsrdb/config/>`_
contains the default configuration for processing each NSRDB variable. Each NSRDB
variable has default scaling factors, data types, physical ranges, etc... The path
specification for this dataset config will be found in the NSRDB code under the `var_meta` kwarg.
If this kwarg is set to None (default), the NSRDB will point to the
`nsrdb_vars.csv <https://github.com/NatLabRockies/nsrdb/tree/main/nsrdb/config/nsrdb_vars.csv>`_ file in this directory.