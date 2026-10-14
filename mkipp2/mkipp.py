"""Create Kippenhahn diagrams from MESA history and profile output.

Requirements: history.data and profiles.data containing
              History (star_age,model_number,star_mass,photosphere_r,mixing_regions,mix_relr_regions)
              Profile (mass,radius,eps_nuc)
"""
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch

import mkipp2.kipp_data as kipp
import mkipp2.mesa_data as md


# Create contour levels array (TBD: Need to be improved)
def get_levels_linear(lmin, lmax, n_levels):
    """Generate evenly spaced contour levels around a data range.

    Parameters
    ----------
    lmin, lmax : float
        Minimum and maximum data values, with ``lmax > lmin``.
    n_levels : int
        Positive number of intervals used to set the spacing.

    Returns
    -------
    numpy.ndarray
        Levels from ``lmin - delta`` up to, but excluding, ``lmax + delta``,
        where ``delta = (lmax - lmin) / n_levels``. Floating-point rounding
        can affect the endpoint and count.
    """
    delta = (lmax - lmin)/n_levels
    levels = np.arange(lmin-delta, lmax+delta, delta)
    return levels


def get_levels_log(lmin, lmax, n_levels):
    """Generate integer contour levels for logarithmic data.

    Parameters
    ----------
    lmin : float
        Minimum logarithmic data value; currently does not affect the result.
    lmax : float
        Maximum logarithmic data value.
    n_levels : int
        Requested number of levels; currently unused.

    Returns
    -------
    range
        Integers from zero through ``int(lmax) + 1``, inclusive. The range
        is empty if its upper bound is negative.

    Notes
    -----
    Input values are already logarithms. This function does not transform
    data or generate logarithmically spaced values in linear units.
    """
    lmax = int(lmax)+1
    lmin = int(round(lmin, 2))-1
    levels = range(0, lmax+1)
    return levels


def default_extractor(identifier, log10_on_data, prof,
                      return_data_columns=False):
    """Extract a profile column or report the column required to read it.

    Parameters
    ----------
    identifier : str
        Profile column name.
    log10_on_data : bool
        If True, return ``log10(abs(data) + 1e-99)``.
    prof : mesa_data.mesa_data or None
        Loaded profile. May be None when requesting column names.
    return_data_columns : bool, optional
        Return required column names instead of data. Default is False.

    Returns
    -------
    numpy.ndarray or list of str
        Extracted values, optionally transformed, or ``[identifier]`` when
        ``return_data_columns`` is True.
    """
    if return_data_columns:
        return [identifier]
    if log10_on_data:
        return np.log10(abs(prof.get(identifier))+1e-99)
    return prof.get(identifier)


class KippArgs:
    """Configure a Kippenhahn plot and load its profile and mixing data.

    Parameters
    ----------
    logs_dirs : sequence of str, optional
        MESA LOGS directories used to discover input files. Defaults to
        ``['LOGS']``.
    profile_paths : sequence of str, optional
        Ordered profile paths. If None or empty, discover them from
        ``profiles.index`` in each LOGS directory.
    history_paths : sequence of str, optional
        History paths in concatenation order. If None or empty, use
        ``history.data`` in each LOGS directory.
    clean_data : bool, optional
        Remove history rows superseded by backups or restarts. Default is
        True.
    extra_history_cols : sequence of str, optional
        Additional history columns to load. Defaults to no extra columns.
    identifier : str, optional
        Quantity identifier passed to the extractor and used as the colorbar
        label. The default extractor treats it as a column name. Defaults
        to ``'eps_nuc'``.
    extractor : callable, optional
        Function called as ``extractor(identifier, log10_on_data, prof)``
        to obtain an array of profile values. It must also accept
        ``prof=None, return_data_columns=True`` and return required column
        names. Defaults to `default_extractor`.
    log10_on_data : bool, optional
        Request logarithmic values from the extractor. Default is True.
        The default extractor computes ``log10(abs(data) + 1e-99)``.
    contour_colormap : matplotlib.colors.Colormap, optional
        Colormap for the filled contours. Defaults to ``'Blues'``.
    levels : array_like, optional
        Explicit, increasing contour levels. If None or empty, generate
        levels from the extracted data.
    log_levels : bool, optional
        Use integer levels for logarithmic data when levels are generated.
        Default is True. This setting does not transform the data.
    num_levels : int, optional
        Number of intervals used by the linear level generator. Defaults
        to 8; the logarithmic level generator currently ignores this value.
    xaxis : {'model_number', 'star_age'}, optional
        Horizontal coordinate. Defaults to ``'model_number'``.
    xaxis_divide : float, optional
        Nonzero divisor applied before the x-axis transform. Defaults to
        1.0. Certain ``time_units`` override it when plotting ``star_age``.
    time_units : {'yr', '1000 yr', 'Myr', 'Gyr'}, optional
        Time-axis label, defaulting to ``'Myr'``. For ``star_age``, the last
        three choices override the divisor with 1000, 1e6, and 1e9,
        respectively. ``'yr'`` retains ``xaxis_divide``.
    function_on_xaxis : callable, optional
        Transform applied after division. Must accept scalar profile ages
        or model numbers and NumPy arrays from histories. Defaults to the
        identity function.
    yaxis : {'mass', 'radius'}, optional
        Vertical coordinate, in solar masses or solar radii. Defaults to
        ``'mass'``.
    yaxis_normalize : bool, optional
        Use a unit surface coordinate for mixing regions and a contour grid
        spanning zero to one. Default is False. See the normalization
        limitation in Notes.
    show_conv, show_therm, show_semi, show_over : bool, optional
        Display convection, thermohaline mixing, semiconvection, and
        overshooting boundaries, respectively. Each defaults to True.
    show_rot : bool, optional
        Display rotational mixing boundaries. Default is False.
    core_masses : sequence of {'He', 'C', 'O'}, optional
        Core masses to overlay when ``yaxis='mass'``. Defaults to all three;
        an empty sequence suppresses them.
    yresolution : int, optional
        Number of vertical interpolation points; must be at least two.
        Defaults to 1000.
    mass_tolerance, radius_tolerance : float, optional
        Minimum mixing-layer thickness as a fraction of the surface
        coordinate, using the tolerance for the selected y axis. Each
        defaults to 1e-7.
    decorate_plot : bool, optional
        Add axis labels and a colorbar. Default is True.
    show_plot : bool, optional
        Call ``pyplot.show()`` after plotting. Default is False.
    save_file : bool, optional
        Save the current pyplot figure after the optional show call.
        Default is True.
    save_filename : str, optional
        Output filename; its extension determines the format. Defaults to
        ``'Kippenhahn.png'``.
    xlims : tuple of float, optional
        Inclusive limits in transformed x coordinates, used for data
        selection and displayed axis limits. Neighboring samples are kept
        for boundary coverage. Defaults to the full range.

    Attributes
    ----------
    xyz_data : namedtuple
        Interpolated profile data with fields ``xlims``, ``X``, ``Y``, and
        ``Z``, returned by `kipp_data.get_xyz_data`.
    mixing_zones : namedtuple
        Mixing-region paths and history data with fields ``zones``,
        ``mix_types``, ``x_coords``, ``y_coords``, and ``histories``, returned
        by `kipp_data.get_mixing_zones`.

    Notes
    -----
    All parameters are keyword-only and are stored as instance attributes,
    with defaults resolved as described above. Construction immediately
    reads the input files and computes both data products; changing input
    settings afterward does not recompute them.

    Normalization is currently incomplete: profile interpolation still uses
    unnormalized mass/radius coordinates, and core-mass overlays remain in
    solar masses even when ``yaxis_normalize`` is True.
    """
    def __init__(self, *,
                 logs_dirs=None,
                 profile_paths=None,
                 history_paths=None,
                 clean_data=True,
                 extra_history_cols=None,
                 identifier="eps_nuc",
                 extractor=default_extractor,
                 log10_on_data=True,
                 contour_colormap=plt.get_cmap("Blues"),
                 levels=None,
                 log_levels=True,
                 num_levels=8,
                 xaxis="model_number",
                 xaxis_divide=1.0,
                 time_units="Myr",
                 function_on_xaxis=lambda x: x,
                 yaxis="mass",
                 yaxis_normalize=False,
                 show_conv=True,
                 show_therm=True,
                 show_semi=True,
                 show_over=True,
                 show_rot=False,
                 core_masses=None,
                 yresolution=1000,
                 mass_tolerance=0.0000001,
                 radius_tolerance=0.0000001,
                 decorate_plot=True,
                 show_plot=False,
                 save_file=True,
                 save_filename="Kippenhahn.png",
                 xlims=None):
        """Store plot settings and prepare profile and mixing-region data."""

        if logs_dirs is None:
            logs_dirs = ['LOGS']

        self.logs_dirs = logs_dirs
        self.clean_data = clean_data

        if extra_history_cols is None:
            extra_history_cols = []
        self.extra_history_cols = extra_history_cols

        self.identifier = identifier
        self.extractor = extractor
        self.log10_on_data = log10_on_data
        self.contour_colormap = contour_colormap

        if levels is None:
            levels = []
        self.levels = levels

        self.log_levels = log_levels
        self.num_levels = num_levels
        self.xaxis = xaxis
        self.xaxis_divide = xaxis_divide
        self.time_units = time_units
        self.function_on_xaxis = function_on_xaxis
        self.yaxis = yaxis
        self.yaxis_normalize = yaxis_normalize
        self.show_conv = show_conv
        self.show_therm = show_therm
        self.show_semi = show_semi
        self.show_over = show_over
        self.show_rot = show_rot

        if core_masses is None:
            core_masses = ["He", "C", "O"]
        self.core_masses = core_masses

        self.yresolution = yresolution
        self.mass_tolerance = mass_tolerance
        self.radius_tolerance = radius_tolerance
        self.decorate_plot = decorate_plot
        self.show_plot = show_plot
        self.save_file = save_file
        self.save_filename = save_filename
        self.xlims = xlims

        # Fill profile and history names if unspecified
        if profile_paths is None:
            profile_paths = []
        self.profile_paths = profile_paths
        if len(self.profile_paths) == 0:
            self.profile_paths = md.get_profile_paths(logs_dirs=self.logs_dirs)

        if history_paths is None:
            history_paths = []
        self.history_paths = history_paths
        if len(self.history_paths) == 0:
            self.history_paths = []
            for log_dir in self.logs_dirs:
                self.history_paths.append(log_dir+"/history.data")

        self.xyz_data = kipp.get_xyz_data(self.profile_paths, self, xlims=xlims)
        self.mixing_zones = kipp.get_mixing_zones(self.history_paths,
                                                  self, xlims=xlims)


def kipp_plot(kipp_args, axis=None):
    """Draw profile contours, mixing boundaries, and stellar structure.

    Parameters
    ----------
    kipp_args : KippArgs
        Prepared data and plot settings. Controls contour levels, visible
        mixing types, core-mass overlays, labels, display, and file output.
    axis : matplotlib.axes.Axes, optional
        Axes on which to draw. If None, create a figure and axes.

    Returns
    -------
    KippPlot
        Named tuple containing ``contour_plot`` (the contour set returned
        by ``Axes.contourf``), ``histories`` (loaded ``mesa_data`` objects),
        and ``xlims`` (the extent of the selected profile coordinates,
        which can extend beyond the requested plotting limits).

    Notes
    -----
    Mixing regions are drawn as unfilled, hatched patches over the profile
    contours. A white fill covers contours above the stellar surface, and
    selected core masses are overlaid for a mass axis. Axis labels and a
    colorbar are added only when ``decorate_plot`` is True.

    Display and file output use pyplot. In particular,
    ``save_file=True`` saves the current pyplot figure, which need not
    be the figure owning a supplied ``axis``. File output occurs after
    the optional show call.

    """

    xaxis_divide = kipp_args.xaxis_divide
    if kipp_args.xaxis == "star_age":
        if kipp_args.time_units == "1000 yr":
            xaxis_divide = 1000
        elif kipp_args.time_units == "Myr":
            xaxis_divide = 1e6
        elif kipp_args.time_units == "Gyr":
            xaxis_divide = 1e9

    if axis is None:
        fig, axis = plt.subplots()
    else:
        fig = axis.figure

    xyz_data = kipp_args.xyz_data

    # only plot if there is data
    if xyz_data.Z.size > 0:
        # Get levels if undefined
        levels = kipp_args.levels
        if len(levels) == 0:
            if kipp_args.log_levels:
                levels = get_levels_log(np.nanmin(xyz_data.Z[:, :]),
                                        np.nanmax(xyz_data.Z[:, :]),
                                        kipp_args.num_levels)
            else:
                levels = get_levels_linear(np.nanmin(xyz_data.Z[:, :]),
                                           np.nanmax(xyz_data.Z[:, :]),
                                           kipp_args.num_levels)
        # make plot
        contour_plot = axis.contourf(xyz_data.X, xyz_data.Y, xyz_data.Z[:, :],
                                     cmap=kipp_args.contour_colormap,
                                     levels=levels, antialiased=False)
    else:
        print("No profile data to plot!")
        contour_plot = axis.contourf([[None, None]], [[None, None]], [[None, None]],
                                     cmap=kipp_args.contour_colormap,
                                     antialiased=False)

    mixing_zones = kipp_args.mixing_zones

    for i, zone in enumerate(mixing_zones.zones):
        color = ""
        # Convective mixing
        if mixing_zones.mix_types[i] == 1 and kipp_args.show_conv:
            color = "Chartreuse"
            hatch = "//"
            line = 1
        # Overshooting
        elif mixing_zones.mix_types[i] == 3 and kipp_args.show_over:
            color = "purple"
            hatch = "x"
            line = 1
        # Semiconvective mixing
        elif mixing_zones.mix_types[i] == 4 and kipp_args.show_semi:
            color = "red"
            hatch = "\\\\"
            line = 1
        # Thermohaline mixing
        elif mixing_zones.mix_types[i] == 5 and kipp_args.show_therm:
            color = "Gold"  # Salmon
            hatch = "||"
            line = 1
        # Rotational mixing
        elif mixing_zones.mix_types[i] == 6 and kipp_args.show_rot:
            color = "brown"
            hatch = "*"
            line = 1
        # Anonymous mixing
        else:
            color = "white"
            hatch = " "
            line = 0
        axis.add_patch(PathPatch(zone, fill=False, hatch=hatch,
                                 edgecolor=color, linewidth=line))

    # limit x_coords to data of contours and add line at stellar surface
    for i, x_coord in enumerate(mixing_zones.x_coords):
        if x_coord > xyz_data.xlims[1]:
            break

    # Also fill with white above the plot to cover any remainder of the
    # contour plot
    axis.fill_between(mixing_zones.x_coords, max(mixing_zones.y_coords),
                      mixing_zones.y_coords, color="w")
    axis.plot(mixing_zones.x_coords[:i], mixing_zones.y_coords[:i], "k-")

    # add core masses
    if kipp_args.yaxis == "mass":
        field_name = ""
        for core_mass in kipp_args.core_masses:
            if core_mass == "He":
                field_name = "he_core_mass"
                color = "b:"
            elif core_mass == "C":
                field_name = "c_core_mass"
                color = "r:"
            elif core_mass == "O":
                field_name = "o_core_mass"
                color = "g:"
            elif core_mass == "CO":
                field_name = "co_core_mass"
                color = "g:"
            elif core_mass == "Fe":
                field_name = "fe_core_mass"
                color = "m:"
            for history in mixing_zones.histories:
                axis.plot(kipp_args.function_on_xaxis(history.get(kipp_args.xaxis) / xaxis_divide),
                          history.get(field_name), color)

    if kipp_args.decorate_plot:
        # add colorbar
        cbar = plt.colorbar(contour_plot, pad=0.05)
        cbar.set_label(kipp_args.identifier)

        # add axis labels
        if kipp_args.xaxis == "star_age":
            axis.set_xlabel(r'$t$ ('+kipp_args.time_units+')')
        else:
            axis.set_xlabel(r'$Model Number$')
        if kipp_args.yaxis == "radius":
            if kipp_args.yaxis_normalize:
                axis.set_ylabel(r'$r/R$')
            else:
                axis.set_ylabel(r'$r/R_\odot$')
        else:
            if kipp_args.yaxis_normalize:
                axis.set_ylabel(r'$m/M$')
            else:
                axis.set_ylabel(r'$m/M_\odot$')

    if kipp_args.xlims is not None:
        axis.set_xlim(kipp_args.xlims)

    if kipp_args.show_plot:
        fig.show()
    if kipp_args.save_file:
        fig.tight_layout()
        fig.savefig(kipp_args.save_filename)

    KippPlot = namedtuple('KippPlot', 'contour_plot histories xlims')

    return KippPlot(contour_plot, mixing_zones.histories, xyz_data.xlims)
