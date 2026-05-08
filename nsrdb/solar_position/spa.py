"""
NREL's solar position algorithm (SPA)

Usage:
with h5py.File(nsrdb_file, 'r') as f:
    time_index = f['time_index'][...] # (t) likely 8760, 17520, 105120
    meta = f['meta'][site_gids]
    lat_lon = meta[['latitude', 'longitude']] # (n, 2)
    elev = meta['elevation'] # (n)
    P = f['pressure'][:, site_gids] # (t, n)
    T = f['air_temperature'][:, site_gids] # (t, n)

apparent_zenith_angle = SPA.zenith(time_index, lat_lon, elev=elev,
                                   pressure=P, temperature=T)
"""

import numpy as np
import pandas as pd

from nsrdb.solar_position.spa_tables import DeltaTable, SPAtables

SPA_TABLES = SPAtables()
DELTA_TABLE = DeltaTable()


class SPA:
    """
    Solar position algorithm
    """

    def __init__(self, time_index, lat_lon, elev=0):
        """
        Parameters
        ----------
        time_index : ndarray | pandas.DatetimeIndex | str
            Datetime stamps of interest
        lat_lon : ndarray
            (latitude, longitude) for site(s) of interest
        elev : ndarray
            Elevation above sea-level for site(s) of interest in meters
        """
        if not isinstance(time_index, pd.DatetimeIndex):
            if isinstance(time_index, str):
                time_index = [time_index]

            time_index = pd.to_datetime(time_index)

        self._time_index = time_index

        if not isinstance(lat_lon, np.ndarray):
            lat_lon = np.array(lat_lon)

        self._lat_lon = np.expand_dims(lat_lon, axis=0).T

        if isinstance(elev, (list, tuple)):
            elev = np.array(elev)

        if isinstance(elev, np.ndarray):
            elev = np.expand_dims(elev, axis=0).T

        self._elev = elev

    @property
    def time_index(self):
        """
        Datetime stamp(s) of interest

        Returns
        -------
        time_index : pandas.DatetimeIndex
        """
        return self._time_index

    @property
    def latitude(self):
        """
        Latitudes of site(s)

        Returns
        -------
        lat : ndarray
        """
        lat = self._lat_lon[0]
        return lat

    @property
    def longitude(self):
        """
        longitude of site(s)

        Returns
        -------
        lon : ndarray
        """
        lon = self._lat_lon[1]
        return lon

    @property
    def altitude(self):
        """
        elevation above sea-level of site(s)
        """
        return self._elev

    def _parse_delta_t(self):
        """Get a delta t value for start of time index

        Returns
        -------
        delta_t : float
            Delta-t value for the first date in the time index.
        """
        date = self.time_index[0].to_pydatetime().date()
        delta_t = DELTA_TABLE[date]
        return delta_t

    def _parse_time(self, delta_t=None):
        """
        Convert UTC datetime index into:
        - Julian day
        - Julian ephemeris day
        - Julian century
        - Julian ephemeris century
        - Julian ephemeris millennium

        Parameters
        ----------
        delta_t : float | None
            Difference between terrestrial time and UT1. Dependent on year.
            None will infer delta_t value from time index (recommended).

        Returns
        -------
        jd : ndarray
            Julian day
        jde : ndarray
            Julian ephemeris day
        jc : ndarray
            Julian century
        jce : ndarray
            Julian ephemeris century
        jme : ndarray
            Julian ephemeris millennium
        """
        if delta_t is None:
            delta_t = self._parse_delta_t()

        jd = self.time_index.to_julian_date().values
        jde = jd + delta_t / 86400
        jc = (jd - 2451545) / 36525
        jce = (jde - 2451545) / 36525
        jme = jce / 10

        return jd, jde, jc, jce, jme

    @staticmethod
    def _helicocentric_vector(arr, jme):
        """
        Perform heliocentric array to vector calculation:

        Parameters
        ----------
        arr : ndarray
            Array to compute vector from
        jme : ndarray
            Julian ephemeris millennium

        Returns
        -------
        out : ndarray
            heliocentric value for each input timestep
        """
        jme = np.expand_dims(jme, axis=0)
        out = np.sum(
            arr[:, :, [0]] * np.cos(arr[:, :, [1]] + arr[:, :, [2]] * jme),
            axis=1,
        )
        out = np.sum(out * np.power(jme.T, range(len(arr))).T, axis=0) / 10**8
        return out

    @staticmethod
    def heliocentric_longitude(jme):
        """
        Compute heliocentric Longitude

        Parameters
        ----------
        jme : ndarray
            Julian ephemeris millennium

        Returns
        -------
        lon : ndarray
            Heliocentric longitude in degrees
        """
        lon = SPA._helicocentric_vector(SPA_TABLES.helio_long_table, jme)
        lon = np.rad2deg(lon) % 360
        return lon

    @staticmethod
    def heliocentric_latitude(jme):
        """
        Compute heliocentric latitude

        Parameters
        ----------
        jme : ndarray
            Julian ephemeris millennium

        Returns
        -------
        b : ndarray
            Heliocentric longitude in degrees
        """
        b = SPA._helicocentric_vector(SPA_TABLES.helio_lat_table, jme)
        b = np.rad2deg(b)
        return b

    @staticmethod
    def heliocentric_radius_vector(jme):
        """
        Compute heliocentric radius

        Parameters
        ----------
        jme : ndarray
            Julian ephemeris millennium

        Returns
        -------
        r : ndarray
            Heliocentric radius in radians
        """
        r = SPA._helicocentric_vector(SPA_TABLES.helio_radius_table, jme)
        return r

    @staticmethod
    def geocentric_longitude(heliocentric_longitude):
        """
        Compute geocentric longitude from heliocentric longitude

        Parameters
        ----------
        heliocentric_longitude : ndarray
            heliocentric longitude for each available timestep

        Returns
        -------
        theta : ndarray
            geocentric longitude for each available timestep
        """
        theta = heliocentric_longitude + 180.0
        return theta % 360

    @staticmethod
    def geocentric_latitude(heliocentric_latitude):
        """
        Compute geocentric latitude from heliocentric Latitude

        Parameters
        ----------
        heliocentric_latitude : ndarray
            heliocentric latitude for each available timestep

        Returns
        -------
        beta : ndarray
            geocentric latitude for each available timestep
        """
        beta = -1.0 * heliocentric_latitude
        return beta

    @staticmethod
    def mean_elongation(jce):
        """
        Compute mean elongation

        Parameters
        ----------
        jce : ndarray
            Julian ephemeris century for all timesteps

        Returns
        -------
        x0 : ndarray
            Mean elogation for all timesteps
        """
        x0 = (
            297.85036
            + 445267.111480 * jce
            - 0.0019142 * jce**2
            + jce**3 / 189474
        )
        return x0

    @staticmethod
    def mean_anomaly_sun(jce):
        """
        Compute mean sun anomaly

        Parameters
        ----------
        jce : ndarray
            Julian ephemeris century for all timesteps

        Returns
        -------
        x1 : ndarray
            Mean sun anomaly for all timesteps
        """
        x1 = (
            357.52772
            + 35999.050340 * jce
            - 0.0001603 * jce**2
            - jce**3 / 300000
        )
        return x1

    @staticmethod
    def mean_anomaly_moon(jce):
        """
        Compute mean moon anomaly

        Parameters
        ----------
        jce : ndarray
            Julian ephemeris century for all timesteps

        Returns
        -------
        x2 : ndarray
            Mean moon anomaly for all timesteps
        """
        x2 = (
            134.96298
            + 477198.867398 * jce
            + 0.0086972 * jce**2
            + jce**3 / 56250
        )
        return x2

    @staticmethod
    def moon_argument_latitude(jce):
        """
        Compute moon latitude

        Parameters
        ----------
        jce : ndarray
            Julian ephemeris century for all timesteps

        Returns
        -------
        x3 : ndarray
            Moon latitude for all timesteps
        """
        x3 = (
            93.27191
            + 483202.017538 * jce
            - 0.0036825 * jce**2
            + jce**3 / 327270
        )
        return x3

    @staticmethod
    def moon_ascending_longitude(jce):
        """
        Compute moon ascending longitude

        Parameters
        ----------
        jce : ndarray
            Julian ephemeris century for all timesteps

        Returns
        -------
        x4 : ndarray
            Moon ascending longitude for all timesteps
        """
        x4 = (
            125.04452
            - 1934.136261 * jce
            + 0.0020708 * jce**2
            + jce**3 / 450000
        )
        return x4

    @staticmethod
    def nutation_coefficients(jce):
        """
        Compute the nutation coefficients:
        x0 = mean elongation
        x1 = mean sun anomaly
        x2 = mean moon anomaly
        x3 = moon latitude
        x4 = moon ascending longitude

        Parameters
        ----------
        jce : ndarray
            Julian ephemeris century for all timesteps

        Returns
        -------
        x_arr : ndarray
            Array of nutation coeffiecients [x0, x1, x2, x3, x4]
        """
        x0 = SPA.mean_elongation(jce)
        x1 = SPA.mean_anomaly_sun(jce)
        x2 = SPA.mean_anomaly_moon(jce)
        x3 = SPA.moon_argument_latitude(jce)
        x4 = SPA.moon_ascending_longitude(jce)
        return np.array([x0, x1, x2, x3, x4])

    @staticmethod
    def nutation_position(jce):
        """
        Compute nutation longitude for all timesteps

        Parameters
        ----------
        jce : ndarray
            Julian ephemeris century for all timesteps

        Returns
        -------
        delta_psi : ndarray
            nutation longitude
        """
        nut_arr = SPA.nutation_coefficients(jce)
        nut_arr = np.radians(np.dot(SPA_TABLES.nutation_yterm_table, nut_arr))

        abcd = SPA_TABLES.nutation_abcd_table
        a = abcd[:, 0]
        b = abcd[:, 1]
        c = abcd[:, 2]
        d = abcd[:, 3]

        argsin = np.sin(nut_arr)
        delta_psi = np.sum(
            (a + b * np.expand_dims(jce, axis=0).T).T * argsin, axis=0
        )
        delta_psi /= 36000000

        argcos = np.cos(nut_arr)
        delta_eps = np.sum(
            (c + d * np.expand_dims(jce, axis=0).T).T * argcos, axis=0
        )
        delta_eps /= 36000000

        return delta_psi, delta_eps

    @staticmethod
    def mean_ecliptic_obliquity(jme):
        """
        Compute mean ecliptic obliquity

        Parameters
        ----------
        jme : ndarray
            Julian ephemeris millennium for all timesteps

        Returns
        -------
        e0 : ndarray
            Mean ecliptic obliquity for all timesteps
        """
        U = np.expand_dims(jme, axis=0).T / 10
        e0_coeff = np.array(
            [
                84381.448,
                -4680.93,
                -1.55,
                1999.25,
                -51.38,
                -249.67,
                -39.05,
                7.12,
                27.87,
                5.79,
                2.45,
            ]
        )

        e0 = np.sum(e0_coeff * np.power(U, range(11)), axis=1)
        return e0

    @staticmethod
    def true_ecliptic_obliquity(e0, delta_eps):
        """
        Compute true ecliptic obliquity

        Parameters
        ----------
        e0 : ndarray
            Mean elciptic obliquity
        delta_eps : ndarray
            Nutation obliquity

        Returns
        -------
        e : ndarray
            True ecliptic obliquity
        """
        e = e0 / 3600 + delta_eps
        return e

    @staticmethod
    def aberration_correction(r):
        """
        Compute aberration correction

        Parameters
        ----------
        r : ndarray
            Heliocentric or earth radius vector

        Returns
        -------
        delta_u : ndarray
            Aberation correction
        """
        delta_u = -20.4898 / (3600 * r)
        return delta_u

    @staticmethod
    def apparent_sun_longitude(beta, delta_psi, delta_u):
        """
        Compute apparent sun longitude

        Parameters
        ----------
        beta : ndarray
            Geocentric latitude
        delta_psi : ndarray
            Nutation longitude
        delta_u : ndarray
            Aberration correction

        Returns
        -------
        lamd : ndarray
            Apparent sun longitude
        """
        lamd = beta + delta_psi + delta_u
        return lamd

    @staticmethod
    def mean_sidereal_time(jd, jc):
        """
        Compute mean sidereal time

        Parameters
        ----------
        jd : ndarray
            Julian day for all timesteps
        jc : ndarray
            Julian century for all timesteps

        v0 : ndarray
            Mean sidereal time in degrees
        """
        v0 = (
            280.46061837
            + 360.98564736629 * (jd - 2451545)
            + 0.000387933 * jc**2
            - jc**3 / 38710000
        )
        return v0 % 360.0

    @staticmethod
    def apparent_sidereal_time(v0, delta_psi, e):
        """
        Compute apparent sidereal time

        Parameters
        ----------
        v0 : ndarray
            Mean sidreal time
        delta_psi : ndarray
            Nutation longitude
        e : ndarray
            True ecliptic obliquity

        Returns
        -------
        v : ndarray
            Apparent sidereal time degrees
        """
        v = v0 + delta_psi * np.cos(np.radians(e))
        return v

    @staticmethod
    def geocentric_sun_position(lamd, e, beta):
        """
        Compute geocentric sun position

        Parameters
        ----------
        lamd : ndarray
            Apparent sun longitude
        e : ndarray
            True ecliptic obliquity
        beta : ndarray
            Geocentric latitude

        Returns
        -------
        alpha : ndarray
            Geocentric sun right ascension in degrees
        delta : ndarray
            Geocentric sun delication in degrees
        """
        lamd = np.radians(lamd)
        e = np.radians(e)
        beta = np.radians(beta)
        num = np.sin(lamd) * np.cos(e) - np.tan(beta) * np.sin(e)
        denom = np.cos(lamd)
        alpha = np.degrees(np.arctan2(num, denom)) % 360

        delta = np.sin(beta) * np.cos(e) + np.cos(beta) * np.sin(e) * np.sin(
            lamd
        )
        delta = np.degrees(np.arcsin(delta))

        return alpha, delta

    @staticmethod
    def local_hour_angle(v, obs_lon, alpha):
        """
        Compute local hour angle measured westward from south

        Parameters
        ----------
        v : ndarray
            Apparent sidreal time
        obs_lon : ndarray
            Observers longitudes
        alpha : ndarray
            Sun right ascension

        Returns
        -------
        H : ndarray
            Local hour angle in degrees from westward from south
        """
        H = v + obs_lon - alpha
        return H % 360

    @staticmethod
    def equatorial_horizontal_parallax(r):
        """
        Computes equatorial horizonatl parallax

        Parameters
        ----------
        r : ndarray
            heliocentric or earth radius vector

        Returns
        -------
        xi : ndarray
            Equatorial horizontal parallax
        """
        xi = 8.794 / (3600 * r)
        return xi

    @staticmethod
    def observer_xy(obs_lat, obs_elev):
        """
        Compute the observer x and y terms

        Parameters
        ----------
        obs_lat : ndarray
            Observers latitudes
        obs_elev : ndarray
            Observers elevations (above sealevel in m)

        Returns
        -------
        obs_x : ndarray
            Observers x terms
        obs_y : ndarray
            Observers y terms
        """
        obs_lat = np.radians(obs_lat)
        u = np.arctan(0.99664719 * np.tan(obs_lat))
        obs_x = np.cos(u) + obs_elev / 6378140 * np.cos(obs_lat)
        obs_y = 0.99664719 * np.sin(u) + obs_elev / 6378140 * np.sin(obs_lat)
        return obs_x, obs_y

    @staticmethod
    def parallax_sun_right_ascension(obs_x, xi, H, delta):
        """
        Computer sun right ascension parallax

        Parameters
        ----------
        obs_x : ndarray
            Observers x terms
        xi : ndarray
            Equatorial horizontal parallax
        H : ndarray
            Local hour angle
        delta : ndarray
            Geocentric sun declination

        Returns
        -------
        delta_alpha : ndarray
            Sun right ascension parallax
        """
        xi = np.radians(xi)
        H = np.radians(H)
        delta = np.radians(delta)
        num = -obs_x * np.sin(xi) * np.sin(H)
        denom = np.cos(delta) - obs_x * np.sin(xi) * np.cos(H)
        delta_alpha = np.degrees(np.arctan2(num, denom))
        return delta_alpha

    @staticmethod
    def topocentric_sun_declination(obs_x, obs_y, xi, H, delta, delta_alpha):
        """
        Compute topocentric sun position: right ascention and declination

        Parameters
        ----------
        obs_x : ndarray
            Observers x terms
        obs_y : ndarray
            Observers y terms
        xi : ndarray
            Equatorial horizontal parallax
        H : ndarray
            Local hour angle
        delta : ndarray
            Geocentric sun declination
        delta_alpha : ndarray
            Sun right ascensoin parallax

        Returns
        -------
        delta_prime : ndarray
            Topocentric sun declination angle in degrees
        """
        # Topocentric sun right ascension angle in degrees
        # alpha_prime = alpha + delta_alpha

        delta = np.radians(delta)
        xi = np.radians(xi)
        H = np.radians(H)
        delta_alpha = np.radians(delta_alpha)
        num = (np.sin(delta) - obs_y * np.sin(xi)) * np.cos(delta_alpha)
        denom = np.cos(delta) - obs_x * np.sin(xi) * np.cos(H)
        delta_prime = np.degrees(np.arctan2(num, denom))
        return delta_prime

    @staticmethod
    def topocentric_local_hour_angle(H, delta_alpha):
        """
        Compute topocentric local hour angle

        Parameters
        ----------
        H : ndarray
            Local hour angle
        delta_alpha : ndarray
            Sun right ascension parallax

        Returns
        -------
        H_prime : ndarray
            Topocentric local hour angle
        """
        H_prime = H - delta_alpha
        return H_prime

    @staticmethod
    def topocentric_solar_position(obs_lat, obs_elev, xi, H, delta):
        """
        Compute the topocentric sun position: elevation and azimuth
        - without atmospheric correction

        Parameters
        ----------
        obs_lat : ndarray
            Observers latitudes
        obs_elev : ndarray
            Observers elevations
        xi : ndarray
            Equatorial horizontal parallax
        H : ndarray
            Local hour angle
        delta : ndarray
            Geocentric sun declination

        Returns
        -------
        e0 : ndarray
            Topocentric elevation angle in degrees
        phi : ndarray
            Topocentric azimuth angle
        """
        obs_x, obs_y = SPA.observer_xy(obs_lat, obs_elev)
        delta_alpha = SPA.parallax_sun_right_ascension(obs_x, xi, H, delta)
        delta_prime = SPA.topocentric_sun_declination(
            obs_x, obs_y, xi, H, delta, delta_alpha
        )
        H_prime = SPA.topocentric_local_hour_angle(H, delta_alpha)

        obs_lat = np.radians(obs_lat)
        delta_prime = np.radians(delta_prime)
        H_prime = np.radians(H_prime)
        e0 = np.sin(obs_lat) * np.sin(delta_prime) + np.cos(obs_lat) * np.cos(
            delta_prime
        ) * np.cos(H_prime)
        e0 = np.degrees(np.arcsin(e0))

        num = np.sin(H_prime)
        denom = np.cos(H_prime) * np.sin(obs_lat) - np.tan(
            delta_prime
        ) * np.cos(obs_lat)
        gamma = np.degrees(np.arctan2(num, denom)) % 360
        phi = (gamma + 180) % 360
        return e0, phi

    @staticmethod
    def _check_shape(arr, shape):
        """
        Check shape of arr and transpose if needed

        Parameters
        ----------
        arr : ndarray
            Array of interest
        shape : tuple
            Desired shape

        Returns
        -------
        arr : ndarray
            arr with proper shape (if possible)
        """
        if len(shape) == 2 and arr.shape[0] == shape[-1]:
            arr = arr.T

        if len(arr.shape) == 1 and arr.shape[0] == shape[1]:
            arr = arr.reshape(shape)

        if arr.shape != shape:
            raise ValueError(
                'Cannot convert array of shape {} to '
                'desired final shape {}'.format(arr.shape, shape)
            )

        return arr

    @staticmethod
    def atmospheric_refraction_correction(
        e0, pres=1013.25, temp=12, atmos_refract=0.5667
    ):
        """
        Compute the atmospheric refraction correction value for all
        sites

        Parameters
        ----------
        e0 : ndarray
            Topocentric elevation angle
        pres : ndarray
            Pressure at all sites in millibars
        temp : ndarray
            Temperature at all sites in C
        atmos_refract : float
            Atmospheric refraction constant

        Returns
        -------
        delta_e : ndarray
            Atmospheric refraction correction
        """
        if isinstance(pres, np.ndarray):
            pres = SPA._check_shape(pres, e0.shape)

        if isinstance(temp, np.ndarray):
            temp = SPA._check_shape(temp, e0.shape)

        # switch sets delta_e when the sun is below the horizon
        switch = e0 >= -1.0 * (0.26667 + atmos_refract)
        angle = np.radians(e0 + 10.3 / (e0 + 5.11))
        delta_e = (
            (pres / 1010.0)
            * (283.0 / (273 + temp))
            * 1.02
            / (60 * np.tan(angle))
        ) * switch
        return delta_e

    @staticmethod
    def apparent_elevation_angle(
        e0, pres=1013.25, temp=12, atmos_refract=0.5667
    ):
        """
        The apparent topocentric elevation angle after refraction

        Parameters
        ----------
        e0 : ndarray
            Topocentric elevation angle
        pres : ndarray
            Pressure at all sites in millibar
        temp : ndarray
            Temperature at all sites in C
        atmos_refract : float
            Atmospheric refraction constant

        Returns
        -------
        e : ndarray
            Apparent topocentric elevation angle after refraction
        """
        a = atmos_refract
        delta_e = SPA.atmospheric_refraction_correction(
            e0, pres=pres, temp=temp, atmos_refract=a
        )
        e = e0 + delta_e
        return e

    @staticmethod
    def topocentric_zenith_angle(e):
        """
        Topocentric zenith angle

        Parameters
        ----------
        e : ndarray
            Topocentric elevation angle

        Returns
        -------
        theta : ndarray
            Topocentric zenith angle
        """
        theta = 90 - e
        return theta

    @staticmethod
    def sun_mean_longitude(jme):
        """
        Compute mean sun longitude for all timesteps

        Parameters
        ----------
        jme : ndarray
            Julian ephemeris millennium

        Returns
        -------
        M : ndarray
            Mean sun longitude
        """
        M = (
            280.4664567
            + 360007.6982779 * jme
            + 0.03032028 * jme**2
            + jme**3 / 49931
            - jme**4 / 15300
            - jme**5 / 2000000
        )
        return M

    @staticmethod
    def equation_of_time(jme, alpha, delta_psi, e):
        """
        Equation of time

        Parameters
        ----------
        jme : ndarray
            Julian ephemeris millennium
        alpha : ndarray
            geocentric sun right ascension
        delta_psi : ndarray
            nutation longitude
        e : ndarray
            True ecliptic obliquity

        Returns
        -------
        E : ndarray
            Equation of time values for all timesteps
        """
        M = SPA.sun_mean_longitude(jme)
        E = M - 0.0057183 - alpha + delta_psi * np.cos(np.radians(e))
        # limit between 0 and 360
        E %= 360
        # convert to minutes
        E *= 4
        greater = E > 20
        less = E < -20
        other = (E <= 20) & (E >= -20)
        E = greater * (E - 1440) + less * (E + 1440) + other * E
        return E

    def _temporal_params(self, delta_t=None):
        """
        Compute the solely time dependant parameters for SPA:
        - Apparent sidereal time (v)
        - Geocentric sun right ascension angle (alpha)
        - Geocentric sun delcination angle (delta)
        - Equatorial horizontal parallax (xi)
        - Equation of time (eot)

        Parameters
        ----------
        delta_t : float | None
            Difference between terrestrial time and UT1. Dependent on year.
            None will infer delta_t value from time index (recommended).

        Returns
        -------
        v : ndarray
            Apparent sidereal time degrees
        alpha : ndarray
            Geocentric sun right ascension in degrees
        delta : ndarray
            Geocentric sun delication in degrees
        xi : ndarray
            Equatorial horizontal parallax in degrees
        eot : ndarray
            Equation of time values for all timesteps
        """
        jd, _, jc, jce, jme = self._parse_time(delta_t=delta_t)
        R = self.heliocentric_radius_vector(jme)
        L = self.heliocentric_longitude(jme)
        B = self.heliocentric_latitude(jme)
        Theta = self.geocentric_longitude(L)
        beta = self.geocentric_latitude(B)
        delta_psi, delta_epsilon = self.nutation_position(jce)
        epsilon0 = self.mean_ecliptic_obliquity(jme)
        epsilon = self.true_ecliptic_obliquity(epsilon0, delta_epsilon)
        delta_tau = self.aberration_correction(R)
        lamd = self.apparent_sun_longitude(Theta, delta_psi, delta_tau)
        v0 = self.mean_sidereal_time(jd, jc)
        v = self.apparent_sidereal_time(v0, delta_psi, epsilon)
        alpha, delta = self.geocentric_sun_position(lamd, epsilon, beta)
        xi = self.equatorial_horizontal_parallax(R)
        eot = self.equation_of_time(jme, alpha, delta_psi, epsilon)

        return v, alpha, delta, xi, eot

    def _elevation_azimuth(self, delta_t=None):
        """
        Compute the solar elevation and azimuth locations and times

        Parameters
        ----------
        delta_t : float | None
            Difference between terrestrial time and UT1. Dependent on year.
            None will infer delta_t value from time index (recommended).

        Returns
        -------
        e0 : ndarray
            Solar elevation in degrees
        phi : ndarray
            Solar azimuth in degrees
        """
        v, alpha, delta, xi, _ = self._temporal_params(delta_t=delta_t)
        H = self.local_hour_angle(v, self.longitude, alpha)
        e0, phi = self.topocentric_solar_position(
            self.latitude, self.altitude, xi, H, delta
        )
        return e0, phi

    def solar_position(self, delta_t=None):
        """
        Compute the solar position for all locations and times:
        - elevation (e0)
        - azimuth (phi)
        - zenith (theta0)

        Parameters
        ----------
        delta_t : float | None
            Difference between terrestrial time and UT1. Dependent on year.
            None will infer delta_t value from time index (recommended).

        Returns
        -------
        e0 : ndarray
            Solar elevation in degrees
        phi : ndarray
            Solar azimuth in degrees
        theta0 : ndarray
            Solar zenith in degrees
        """
        # compute temporal params
        e0, phi = self._elevation_azimuth(delta_t=delta_t)
        theta0 = self.topocentric_zenith_angle(e0)
        return e0.T, phi.T, theta0.T

    def apparent_solar_position(
        self,
        pressure=1013.25,
        temperature=12,
        atmospheric_refraction=0.5667,
        delta_t=None,
    ):
        """
        Compute the apparent (atmospheric refraction corrected) solar position
        for all locations and times:
        - elevation (e)
        - zenith (theta)

        Parameters
        ----------
        pressure : ndarray
            Pressure at all sites in millibar
        temperature : ndarray
            Temperature at all sites in C
        atmospheric_refract : float
            Atmospheric refraction constant
        delta_t : float | None
            Difference between terrestrial time and UT1. Dependent on year.
            None will infer delta_t value from time index (recommended).

        Returns
        -------
        e : ndarray
            Solar elevation after atmospheric refraction correction in degrees
        theta : ndarray
            Solar zenith after atmospheric refraction correction in degrees
        """
        e0, _ = self._elevation_azimuth(delta_t=delta_t)
        e = self.apparent_elevation_angle(
            e0,
            pres=pressure,
            temp=temperature,
            atmos_refract=atmospheric_refraction,
        )
        theta = self.topocentric_zenith_angle(e)
        return e.T, theta.T

    @classmethod
    def position(cls, time_index, lat_lon, elev=0, delta_t=None):
        """
        Compute the solar position:
        - elevation
        - azimuth
        - zenith

        Parameters
        ----------
        time_index : ndarray | pandas.DatetimeIndex | str
            Datetime stamps of interest
        lat_lon : ndarray
            (latitude, longitude) for site(s) of interest
        elev : ndarray
            Elevation above sea-level for site(s) of interest
        delta_t : float | None
            Difference between terrestrial time and UT1. Dependent on year.
            None will infer delta_t value from time index (recommended).

        Returns
        -------
        e0 : ndarray
            Solar elevation in degrees
        phi : ndarray
            Solar azimuth in degrees
        theta0 : ndarray
            Solar zenith in degrees
        """
        spa = cls(time_index, lat_lon, elev=elev)
        e0, phi, theta0 = spa.solar_position(delta_t=delta_t)
        return e0, phi, theta0

    @classmethod
    def elevation(cls, time_index, lat_lon, elev=0, delta_t=None):
        """
        Compute the solar elevation

        Parameters
        ----------
        time_index : ndarray | pandas.DatetimeIndex | str
            Datetime stamps of interest
        lat_lon : ndarray
            (latitude, longitude) for site(s) of interest
        elev : ndarray
            Elevation above sea-level for site(s) of interest
        delta_t : float | None
            Difference between terrestrial time and UT1. Dependent on year.
            None will infer delta_t value from time index (recommended).

        Returns
        -------
        e0 : ndarray
            Solar elevation in degrees
        """
        spa = cls(time_index, lat_lon, elev=elev)
        e0, _, _ = spa.solar_position(delta_t=delta_t)
        return e0

    @classmethod
    def azimuth(cls, time_index, lat_lon, elev=0, delta_t=None):
        """
        Compute the solar azimuth

        Parameters
        ----------
        time_index : ndarray | pandas.DatetimeIndex | str
            Datetime stamps of interest
        lat_lon : ndarray
            (latitude, longitude) for site(s) of interest
        elev : ndarray
            Elevation above sea-level for site(s) of interest
        delta_t : float | None
            Difference between terrestrial time and UT1. Dependent on year.
            None will infer delta_t value from time index (recommended).

        Returns
        -------
        phi : ndarray
            Solar azimuth in degrees
        """
        spa = cls(time_index, lat_lon, elev=elev)
        _, phi, _ = spa.solar_position(delta_t=delta_t)
        return phi

    @classmethod
    def zenith(cls, time_index, lat_lon, elev=0, delta_t=None):
        """
        Compute the solar zenith

        Parameters
        ----------
        time_index : ndarray | pandas.DatetimeIndex | str
            Datetime stamps of interest
        lat_lon : ndarray
            (latitude, longitude) for site(s) of interest
        elev : ndarray
            Elevation above sea-level for site(s) of interest
        delta_t : float | None
            Difference between terrestrial time and UT1. Dependent on year.
            None will infer delta_t value from time index (recommended).

        Returns
        -------
        theta0 : ndarray
            Solar zenith in degrees
        """
        spa = cls(time_index, lat_lon, elev=elev)
        _, _, theta = spa.solar_position(delta_t=delta_t)
        return theta

    @classmethod
    def apparent_position(
        cls,
        time_index,
        lat_lon,
        elev=0,
        pressure=1013.25,
        temperature=12,
        atmospheric_refraction=0.5667,
        delta_t=None,
    ):
        """
        Compute the solar position after atmospheric refraction correction
        - elevation
        - zenith

        Parameters
        ----------
        time_index : ndarray | pandas.DatetimeIndex | str
            Datetime stamps of interest
        lat_lon : ndarray
            (latitude, longitude) for site(s) of interest
        elev : ndarray
            Elevation above sea-level for site(s) of interest
        pressure : ndarray
            Pressure at all sites in millibar
        temperature : ndarray
            Temperature at all sites in C
        atmospheric_refract : float
            Atmospheric refraction constant
        delta_t : float | None
            Difference between terrestrial time and UT1. Dependent on year.
            None will infer delta_t value from time index (recommended).

        Returns
        -------
        e : ndarray
            Solar elevation after atmospheric refraction correction in degrees
        theta : ndarray
            Solar zenith after atmospheric refraction correction in degrees
        """
        spa = cls(time_index, lat_lon, elev=elev)
        a = atmospheric_refraction
        e, theta = spa.apparent_solar_position(
            pressure=pressure,
            temperature=temperature,
            atmospheric_refraction=a,
            delta_t=delta_t,
        )
        return e, theta

    @classmethod
    def apparent_elevation(
        cls,
        time_index,
        lat_lon,
        elev=0,
        pressure=1013.25,
        temperature=12,
        atmospheric_refraction=0.5667,
        delta_t=None,
    ):
        """
        Compute the solar elevation after atmospheric refraction correction

        Parameters
        ----------
        time_index : ndarray | pandas.DatetimeIndex | str
            Datetime stamps of interest
        lat_lon : ndarray
            (latitude, longitude) for site(s) of interest
        elev : ndarray
            Elevation above sea-level for site(s) of interest
        pressure : ndarray
            Pressure at all sites in millibar
        temperature : ndarray
            Temperature at all sites in C
        atmospheric_refract : float
            Atmospheric refraction constant
        delta_t : float | None
            Difference between terrestrial time and UT1. Dependent on year.
            None will infer delta_t value from time index (recommended).

        Returns
        -------
        e : ndarray
            Solar elevation after atmospheric refraction correction in degrees
        """
        spa = cls(time_index, lat_lon, elev=elev)
        a = atmospheric_refraction
        e, _ = spa.apparent_solar_position(
            pressure=pressure,
            temperature=temperature,
            atmospheric_refraction=a,
            delta_t=delta_t,
        )
        return e

    @classmethod
    def apparent_zenith(
        cls,
        time_index,
        lat_lon,
        elev=0,
        pressure=1013.25,
        temperature=12,
        atmospheric_refraction=0.5667,
        delta_t=None,
    ):
        """
        Compute the solar zenith angle after atmospheric refraction correction

        Parameters
        ----------
        time_index : ndarray | pandas.DatetimeIndex | str
            Datetime stamps of interest
        lat_lon : ndarray
            (latitude, longitude) for site(s) of interest
        elev : ndarray
            Elevation above sea-level for site(s) of interest
        pressure : ndarray
            Pressure at all sites in millibar
        temperature : ndarray
            Temperature at all sites in C
        atmospheric_refract : float
            Atmospheric refraction constant
        delta_t : float | None
            Difference between terrestrial time and UT1. Dependent on year.
            None will infer delta_t value from time index (recommended).

        Returns
        -------
        theta : ndarray
            Solar zenith after atmospheric refraction correction in degrees
        """
        spa = cls(time_index, lat_lon, elev=elev)
        a = atmospheric_refraction
        _, theta = spa.apparent_solar_position(
            pressure=pressure,
            temperature=temperature,
            atmospheric_refraction=a,
            delta_t=delta_t,
        )
        return theta
