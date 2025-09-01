"""
4/18/2025

Script with several postprocessing functions for SAM MSE output

combine_MSE_output(case): combines 2D and 3D chunked MSE output files for a given case
create_pcoord_output(case, avg_period=150, tstart=0, tend=0): creates a time-resolved dataset in pressure coordinates for a given case
create_time_averaged_output(case, avg_period=150, tstart=0, tend=0): creates a time-averaged MSE dataset for a given case
combine_time_averaged_output(fnames, new_file_name): combines the time-averaged output files the different cases (SST, qflux, co2, etc) in a single group
"""

import xarray as xr
import numpy as np
import sam
import sat
import constants as const
import os
import multiprocessing
import re
from functools import partial
import warnings
warnings.filterwarnings('ignore')
SSTs = np.arange(298, 310, 2)

# SSTs = []
# Specify the number of processors to use
ncpu = 1

# Specify the cases to be processed
cases = []



cases += ['fixed_{}d4_nx3072'.format(int(sst))
          for sst in SSTs]
cases += ['fixed_{}d3_nx3072'.format(int(sst))
          for sst in SSTs]

cases += ['slabmw_{}d4_ds100_nx3072'.format(int(sst))
          for sst in SSTs]

cases += ['slabmw_{}d4_ds70_nx3072'.format(int(sst))
          for sst in SSTs]

cases += ['slabmw_{}d4_ds20_nx3072'.format(int(sst))
          for sst in SSTs]
"""
cases += ['fixed_{}d2_nx3072'.format(int(sst))
          for sst in SSTs]
cases += ['slabf_{}d4_nx3072'.format(int(sst))
          for sst in SSTs]



cases += ['slabmw_{}d4_ds35_nx3072'.format(int(sst))
          for sst in SSTs]
cases += ['slabmw_{}d4_ds40_nx3072'.format(int(sst))
          for sst in SSTs]
cases += ['slabmw_{}d4_ds50_nx3072'.format(int(sst))
          for sst in SSTs]
cases += ['slabmw_{}d4_ds60_nx3072'.format(int(sst))
          for sst in SSTs]


cases += ['fixed3d_{}d4_nx3072'.format(int(sst))
          for sst in SSTs]

cases += ['slabf3d_{}d4_nx3072'.format(int(sst))
          for sst in SSTs]

cases += ['slabmw3d_{}d4_ds70_nx3072'.format(int(sst))
          for sst in SSTs]

cases += ['fixed_{}d4_nx1536'.format(int(sst))
          for sst in SSTs]

cases += ['slabmw_{}d4_ds70_nx1536'.format(int(sst))
          for sst in SSTs]


cases += ['fixed_{}d4'.format(int(sst))
          for sst in SSTs[:1]]

cases += ['slabmw_{}d4_ds70'.format(int(sst))
          for sst in SSTs[:1]]

cases += ['fixed_{}d8_nx3072'.format(int(sst))
          for sst in SSTs]

cases += ['fixed_{}d8_nx6144'.format(int(sst))
          for sst in SSTs]

cases += ['fixed_{}d6_nx4608'.format(int(sst))
          for sst in SSTs]

cases += ['fixed_{}d2_nx1536'.format(int(sst))
          for sst in SSTs]

cases += ['slabmw_{}d2_ds35_nx1536'.format(int(sst))
          for sst in SSTs]

cases += ['slabmw_{}d6_ds105_nx4608'.format(int(sst))
          for sst in ['304', '308']]

cases += ['slabmw_{}d4_ds70_nx4608'.format(int(sst))
          for sst in SSTs]
"""
# Specify start and stop time for the time-averaged dataset. Either:
# 1. The last N days to be averaged (avg_period > 0, tstart and tend ignored)
# 2. Specific start and end times (avg_period = 0, tstart and tend used)
avg_period = 0  # set to 0 if you want to specify start and end times
tstart = 150  # ignored if avg_period > 0
tend = 300  # ignored if avg_period > 0


DIR_2D = '/home/as3675/walker/OUT_2D'
DIR_3D = '/home/as3675/walker/OUT_3D'
DIR_MSE = '/home/as3675/walker/OUT_MSE'
DIR_MSE_P = '/home/as3675/walker/OUT_MSE/FULL_PCOORDS'
DIR_MEAN_MSE = '/home/as3675/walker/OUT_MSE/MEAN'
DIR_MEAN_MSE_COMBINED = '/home/as3675/walker/OUT_MSE/MEAN/COMBINED_CASES'


def combine_MSE_output(case):
    """
    Combine the 2D and 3D MSE output files for a given case.
    Saves the combined output to a netCDF file in DIR_MSE
    """

    print(case)

    # Pick the case ID out of the case string
    if case.startswith('p3_'):
        isplit = case.find('_', case.find('_') + 1)
        case_dir = case[:isplit]
    else:
        isplit = case.find('_')
        case_dir = case[:isplit]
    caseID = case[isplit:]

    # Output file names
    f2d = f'{DIR_2D}/{case_dir}_MSE{caseID}_0096.2Dbin_1.nc'
    f3d = f'{DIR_3D}/{case_dir}_MSE{caseID}_0096.bin2D_1.nc'

    # Load the 2D and 3D datasets
    mse2d = xr.load_dataset(f2d).drop_duplicates('time')
    mse3d = xr.load_dataset(f3d).drop_duplicates('time')
    print('  Loaded')

    # Rename 2D variables that have the same name as their 3D counterparts
    rename_dict = {}
    for varname in mse2d.keys():
        if varname in mse3d.keys():
            rename_dict[varname] = f'{varname}2D'
    mse2d = mse2d.rename(rename_dict)

    # Combine datasets
    ds = xr.merge([mse2d, mse3d])

    # Get a few variables before we convert to pressure coordinates
    ds['PRESSURE'] = ds.p + ds.PP/100  # pressure in hPa
    ds['PSFC'] = ds.PRESSURE.isel(
        z=0) + const.g*ds.z.values[0]*ds.RHO.isel(z=0)/100

    ds = sam.add.omega(ds)
    ds['QSAT'] = sat.wsat(ds.TABS, ds.PRESSURE*100)
    ds['SPHUM'] = ds.QV / (1 + ds.QV)
    ds['SPHUMSAT'] = sat.qsat(ds.TABS, ds.PRESSURE*100)
    ds['RH'] = ds.SPHUM / ds.SPHUMSAT
    ds['MSE'] = ds.TABS + const.Lv*ds.SPHUM/const.cp + const.g*ds.z/const.cp
    ds['MSESAT'] = ds.TABS + const.Lv * \
        ds.SPHUMSAT/const.cp + const.g*ds.z/const.cp

    # Function to interpolate a variable profile to a specified pressure level
    def interp_to_pressure(pressure, var_profile, target_pressure):
        return np.interp(np.log(target_pressure), np.log(pressure[::-1]), var_profile[::-1])

    ufunc_args = {
        'input_core_dims': [['z'], ['z'], []],
        'output_core_dims': [[]],
        'vectorize': True,
        'dask': 'parallelized',
        'output_dtypes': [float]
    }

    target_levels = [500, 600, 700, 850]
    for level in target_levels:
        ds[f'Z{level}'] = xr.apply_ufunc(
            interp_to_pressure, ds.PRESSURE, ds.z, level, **ufunc_args)
        ds[f'T{level}'] = xr.apply_ufunc(
            interp_to_pressure, ds.PRESSURE, ds.TABS, level, **ufunc_args)
        ds[f'SPHUMSAT{level}'] = sat.qsat(ds[f'Z{level}'], level*100)
        ds[f'MSESAT{level}'] = ds[f'T{level}'] + const.Lv * \
            ds[f'SPHUMSAT{level}']/const.cp + const.g*ds[f'Z{level}']/const.cp

    ds['DRAD'] = ds.DRADLW + ds.DRADSW
    ds['DRADCL'] = ds.DRADLWCL + ds.DRADSWCL

    ds['RTOA'] = ds.SWNT - ds.LWNT
    ds['RTOACL'] = ds.SWNTC - ds.LWNTC
    ds['RSFC'] = ds.SWNS - ds.LWNS

    # TOA cloud radiative effects
    ds['SWCRE'] = ds.SWNT - ds.SWNTC
    ds['LWCRE'] = ds.LWNTC - ds.LWNT
    ds['NCRE'] = ds.SWCRE + ds.LWCRE

    # atmospheric cloud radiative effects
    ds['ASWCRE'] = ds.DRADSW-ds.DRADSWCL
    ds['ALWCRE'] = ds.DRADLW-ds.DRADLWCL
    ds['ANCRE'] = ds.DRAD-ds.DRADCL

    # surface cloud radiative effects
    ds['SWCRESFC'] = ds.SWNS - ds.SWNSC
    ds['LWCRESFC'] = ds.LWNSC - ds.LWNS
    ds['NCRESFC'] = ds.SWCRESFC + ds.LWCRESFC

    # MSE FLUXES
    ds['FMSEHFLUX'] = ds.FMSE * ds.U  # J/kg m/s
    ds['FMSEVFLUX'] = ds.FMSE * ds.OMEGA  # J/kg m/s

    # 2d column-integrated FMSE advection
    ds['FMSEADV2D'] = ds.FMSEEDDY2D - ds.LHF - ds.SHF

    print('Saving')
    fname = f'{DIR_MSE}/{case}.nc'
    if os.path.exists(fname):
        os.remove(fname)
    ds.to_netcdf(fname)
    print('Finished {}'.format(case))

    ds.close()
    mse2d.close()
    mse3d.close()


def create_pcoord_output(case, avg_period=150, tstart=0, tend=0):
    """
    Creates a time-resolved dataset in pressure coordinates for a given case.
    The time period is defined by tstart and tend, or by avg_period if it is greater than 0.
    Args:
         - case (str)
         avg_period(int): If greater than 0, the last N days will be averaged.
         tstart(int): Start time for the averaging period(ignored if avg_period > 0).
         tend(int): End time for the averaging period(ignored if avg_period > 0).
    Saves the output to a netCDF file in DIR_MSE_P.
    """

    dd = xr.open_dataset(os.path.join(DIR_MSE, case+'.nc'))

    # drop stuff we don't need
    droplist = [
        'QTOTSTOR', 'QTOTEDDY', 'QTOTLSF', 'QTOTMISC', 'QTOTSED', 'QICESED',
        'SLISTOR', 'SLIEDDY', 'SLILSF', 'SLIMISC', 'SLISED',
        'SLISTOR2D', 'SLIEDDY2D', 'SLILSF2D', 'SLIMISC2D', 'SLISED2D',
        'QTOTFLUX', 'QVFLUX', 'QCLFLUX', 'QCIFLUX', 'QPLFLUX', 'QPIFLUX',
        'RHOWU', 'RHOWV', 'RHOWW',  # cutoff 6/2/25
        'PRECICE', 'V', 'BUOYFLUX', 'SLIFLUX', 'FMSELSF'

    ]
    dd = dd.drop_vars([k for k in droplist if k in dd])

    # Get proper time period
    if avg_period > 0:
        t1 = dd.time.max().values
        t0 = t1 - avg_period
    else:
        t0 = tstart
        t1 = tend
    tmpp = dd.sel(time=slice(t0, t1))

    # switch to pressure coordinates using mean pressure profile
    z = tmpp.z.values
    tmpp['z'] = (('z'), tmpp.p.values)
    tmpp = tmpp.drop('p').rename({'z': 'p'})
    tmpp['p'].attrs = {'long_name': 'p', 'units': 'hPa'}
    tmpp['z'] = (('p'), z, {'long_name': 'z', 'units': 'm'})

    pgrid = np.concatenate(([1007], np.arange(1000, 60, -10)))
    tmpp = tmpp.interp(p=pgrid)

    # Harmonize fill values before saving
    for var in tmpp.variables:
        if hasattr(tmpp[var], 'encoding'):
            if '_FillValue' in tmpp[var].encoding and 'missing_value' in tmpp[var].encoding:
                # Keep _FillValue and remove missing_value
                tmpp[var].encoding.pop('missing_value', None)

    # Save
    fname = f'{case}.nc'
    fpath = os.path.join(DIR_MSE_P, fname)
    if os.path.exists(fpath):
        os.remove(fpath)
    tmpp.to_netcdf(fpath)

    # Clean up
    dd.close()
    tmpp.close()
    print(fname)
    return


def create_time_averaged_output(case, avg_period=150, tstart=0, tend=0):
    """
    Creates a time-averaged MSE dataset for a given case.
    The time period is defined by tstart and tend, or by avg_period if it is greater than 0.
    The output is saved to a netCDF file in DIR_MEAN_MSE.
    """

    dd = xr.open_dataset(os.path.join(DIR_MSE, case+'.nc'))

    # drop stuff we don't need
    droplist = [
        'QTOTSTOR', 'QTOTEDDY', 'QTOTLSF', 'QTOTMISC', 'QTOTSED', 'QICESED',
        'SLISTOR', 'SLIEDDY', 'SLILSF', 'SLIMISC', 'SLISED',
        'SLISTOR2D', 'SLIEDDY2D', 'SLILSF2D', 'SLIMISC2D', 'SLISED2D',
        'QTOTFLUX', 'QVFLUX', 'QCLFLUX', 'QCIFLUX', 'QPLFLUX', 'QPIFLUX',
        'RHOWU', 'RHOWV', 'RHOWW',  # cutoff 6/2/25
        'PRECICE', 'V', 'BUOYFLUX', 'SLIFLUX', 'FMSELSF'

    ]
    dd = dd.drop_vars([k for k in droplist if k in dd])

    ################################
    # Before time averaging, add SST statistics that need time-resolved output
    dd['SSTrange'] = dd.SST.max('x') - dd.SST.min('x')
    dd['SSTwarm20'] = dd.SST.where(
        dd.SST >= dd.SST.quantile(0.8, dim='x')).mean('x')
    dd['SSTcold20'] = dd.SST.where(
        dd.SST <= dd.SST.quantile(0.2, dim='x')).mean('x')
    dd['SSTdif'] = dd.SSTwarm20 - dd.SSTcold20
    dd = dd.drop_vars('quantile')

    ################################
    # Time average
    if avg_period > 0:
        t1 = dd.time.max().values
        t0 = t1 - avg_period
        tmpp = dd.sel(time=slice(t0, t1)).mean('time')
    else:
        t0 = tstart
        t1 = tend
        tmpp = dd.sel(time=slice(t0, t1)).mean('time')

    ################################
    # Get local energy storage term for slab ocean (will come out to 0 for fixed SST cases)
    # isntantaneous output for the time period that was averaged
    tslice = dd.sel(time=slice(t0, t1))
    # change in SST(x) over time period (K)
    delta_SST = tslice.SST.isel(time=-1) - tslice.SST.isel(time=0)
    delta_t = (tslice.time.values[-1] -
               # length of averaging period (seconds)
               tslice.time.values[0]) * 86400
    rhow = 1000  # kg/m3, density of water used by SAM
    cpw = 4187  # J/kg/K, specific heat of water used by SAM
    mld = 10  # m, mixed layer depth
    tmpp['MLSTOR'] = delta_SST * rhow * cpw * \
        mld / delta_t  # W/m2, mixed layer storage

    ################################
    # Transform to pressure coordinates
    # To be as exact as possible, we will use the perturbation pressure at each grid cell and
    # individually interpolate each column onto a common pressure grid
    p_nat = tmpp.p.values     # mean pressure profile
    z_nat = tmpp.z.values     # height levels
    # target pressure grid in hPa
    pgrid = np.concatenate(([1007], np.arange(1000, 10, -10)))
    tmpp = tmpp.drop('p')
    tmp = xr.apply_ufunc(  # interpolates each column onto target pressure grid
        lambda data, p: np.interp(
            np.log(pgrid[::-1]), np.log(p[::-1]), data[::-1])[::-1],
        tmpp,
        tmpp.PRESSURE,
        input_core_dims=[["z"], ["z"]],
        output_core_dims=[["pgrid"]],
        exclude_dims=set(("z",)),
        on_missing_core_dim='copy',
        vectorize=True,
        dask='parallelized',
        output_dtypes=[tmpp.PRESSURE.dtype],
        output_sizes={"pgrid": len(pgrid)},
    ).rename({'pgrid': 'p'}).assign_coords({'p': pgrid})
    tmp['p'].attrs = {'long_name': 'p', 'units': 'hPa'}

    tmp['z'] = (('p'), np.interp(  # interpolates native height levels onto target pressure grid
        np.log(pgrid[::-1]), np.log(p_nat[::-1]), z_nat[::-1])[::-1])
    tmp['z'].attrs = {'long_name': 'z', 'units': 'm'}

    # Calculate pressure thickness of each vertical level -- 10 hPa everywhere except for the first two levels
    dp = np.full_like(tmp.TABS, 10)  # Initialize with 10 hPa everywhere
    # use surface pressure to calculate dp for the first level
    dp[:, 0] = tmp.PSFC - (tmp.p.values[0]+tmp.p.values[1])/2
    dp[:, 1] = 8.5  # dp for level centered at 1000 hPa extending from 1003.5 to 995
    tmp['dp'] = (('x', 'p'), dp)  # Add dp to the dataset
    tmp['dp'].attrs = {
        'long_name': 'pressure thickness of each vertical level', 'units': 'hPa'}

    ################################
    # Add statistics to time-averaged output

    # First the streamfunction, which we will use to locate tropopause
    tmp['psi'] = (tmp.U * tmp.dp * 100 / const.g).cumsum('p')  # kg/m/s
    tmp['psi'].attrs = {'long_name': 'streamfunction', 'units': 'kg/m/s'}

    # tropopause - approximated as the pressure where the streamfunction goes to zero (or close to it)
    mean_psi = abs(tmp.psi).mean('x').compute()
    # try:
    tmp['p_top'] = tmp.p.where((tmp.p < 220) & (
        mean_psi < 200), drop=True).max('p') - 1
    # except: # find rel min of streamfunction
    #    tmp['p_top'] = mean_psi.sel(p=slice(220,100)).idxmin('p')

    # Ascent region statistics based on column-averaged omega
    tmp['col_OMEGA'] = tmp.OMEGA.sel(p=slice(1015, tmp.p_top)).weighted(
        tmp.dp.sel(p=slice(1015, tmp.p_top))).mean('p')
    tmp['up'] = (tmp.col_OMEGA < 0)  # .compute()
    tmp['dn'] = (tmp.col_OMEGA > 0)  # .compute()
    tmp['sigma_c'] = tmp.up.mean('x')
    tmp['sigma_n'] = tmp.dn.mean('x')
    tmp['OMEGA_c'] = tmp.col_OMEGA.where(tmp.up).mean('x')
    tmp['OMEGA_n'] = tmp.col_OMEGA.where(tmp.dn).mean('x')

    # Ascent region statistics based on precipitation thresholds (90% and 95% of total precip)
    # Starting with the rainiest column, we add the next rainiest column to the ascent region
    # until the ascent region accounts for more than 90%/95% of the total precipitation
    def find_precip_threshold(precip, frac_total=0.9):
        """
        Find the threshold for chunk-avg precipitation that accounts for a given fraction of the total precipitation.
        """
        precip_sorted = np.sort(precip)[::-1]  # sorted high to low
        precip_threshold = precip_sorted[np.where(
            (np.cumsum(precip_sorted)/precip.sum()) > frac_total)[0][0]]
        return precip_threshold

    tmp['prec_threshold90'] = xr.apply_ufunc(  # finds precip value for which all rainier columns account for 90% of total precip
        find_precip_threshold, tmp.PREC, 0.90, input_core_dims=[['x'], []],
        output_core_dims=[[]], vectorize=True, dask='parallelized',
        output_dtypes=[float])
    # mask indicating rainy region
    tmp['wet90'] = tmp.PREC > tmp.prec_threshold90
    tmp['sigma_p90'] = tmp.wet90.mean('x')  # rainy region area fraction

    tmp['prec_threshold95'] = xr.apply_ufunc(  # finds precip value for which all rainier columns account for 95% of total precip
        find_precip_threshold, tmp.PREC, 0.95, input_core_dims=[['x'], []],
        output_core_dims=[[]], vectorize=True, dask='parallelized',
        output_dtypes=[float])
    # mask indicating rainy region
    tmp['wet95'] = tmp.PREC > tmp.prec_threshold95
    tmp['sigma_p95'] = tmp.wet95.mean('x')  # rainy region area fraction

    # total surface turbulent fluxes
    tmp['HF'] = tmp.LHF + tmp.SHF

    # FMSE advective tendencies
    # FMSE tendency due to all atmospheric motions (W/m2)
    # tendency due to all advective terms (eddy + mean flow) W/m2
    tmp['FMSEADV2D'] = tmp.FMSEEDDY2D - tmp.HF
    # Vertical advection by the mean flow (K/day)
    tmp['FMSEVADV'] = -86400 * tmp.OMEGA * tmp.FMSE.differentiate('p')/100
    # Horizontal advection by the mean flow (K/day)
    tmp['FMSEHADV'] = -86400 * tmp.U * tmp.FMSE.differentiate('x')
    # Total advective tendency by the mean flow (K/day)
    tmp['FMSEMADV'] = tmp.FMSEVADV + tmp.FMSEHADV

    # Vertically integrated FMSE advective tendencies (W/m2)
    # (negative sign b/c pressure goes from high to low in dataset)
    cp = const.cp  # J/kg/K
    g = const.g  # m/s2
    tmp['FMSEVADV2D'] = -(tmp.FMSEVADV*tmp.dp*cp/g).where(tmp.p >=
                                                          tmp.p_top).sum('p')*100/86400
    tmp['FMSEHADV2D'] = -(tmp.FMSEHADV*tmp.dp*cp/g).where(tmp.p >=
                                                          tmp.p_top).sum('p')*100/86400
    tmp['FMSEMADV2D'] = -(tmp.FMSEMADV*tmp.dp*cp/g).where(tmp.p >=
                                                          tmp.p_top).sum('p')*100/86400
    tmp['FMSEEADV2D'] = tmp.FMSEADV2D - tmp.FMSEMADV2D

    # water vapor advective tendency
    tmp['QTOTADV2D'] = tmp.QTOTEDDY2D - tmp.LHF/const.Lv  # kg/m2/s

    # Adjust radiation fields to more closely reflect top-of-model rather than TOA so budget is closed
    tmp = tmp.rename({'RTOA': 'RTOA_orig'})
    tmp['LWNT'] = tmp.LWNS - tmp.DRADLW
    tmp['SWNT'] = tmp.SWNS + tmp.DRADSW
    tmp['LWNTC'] = tmp.LWNSC - tmp.DRADLWCL
    tmp['SWNTC'] = tmp.SWNSC + tmp.DRADSWCL
    tmp['RTOA'] = tmp.SWNT - tmp.LWNT
    tmp['RTOACL'] = tmp.SWNTC - tmp.LWNTC
    tmp['SWCRE'] = tmp.SWNT - tmp.SWNTC
    tmp['LWCRE'] = tmp.LWNTC - tmp.LWNT
    tmp['NCRE'] = tmp.SWCRE + tmp.LWCRE

    # GMS as the total MSE tendency due to atmospheric motions (including eddy) normalized by vertical velocity
    # tmp['GMS'] = tmp.FMSEADV2D.where(tmp.up).mean(
    #    'x') / tmp.col_OMEGA.where(tmp.up).mean('x')

    # statistics averaged over the boundary layer -- here defined as 950 hPa to surface
    def pbl_avg(var): return tmp[var].sel(p=slice(1015, 950)).weighted(
        tmp.dp.sel(p=slice(1015, 950))).mean('p')
    tmp['Tbl'] = pbl_avg('TABS')
    tmp['SPHUMbl'] = pbl_avg('SPHUM')
    tmp['SPHUMSATbl'] = pbl_avg('SPHUMSAT')
    tmp['RHbl'] = pbl_avg('RH')
    tmp['MSEbl'] = pbl_avg('MSE')
    tmp['MSESATbl'] = pbl_avg('MSESAT')

    # S (ocean heat source) if applicable
    # To get the correct chunk-averaged values, we will first define S on the native grid, then chunk avg
    if case.startswith('slabmw'):
        dS = float(re.search(r'_ds(\d{2})', case).group(1))
        Smatch = re.search(r'd4_s(\d{2,3})', case)
        if Smatch:
            num_str = Smatch.group(1)
            S = float(num_str) if len(num_str) == 2 else float(
                num_str[0:2]+'.'+num_str[2])
        else:
            S = 0

        # original x grid (before chunk averaging)
        dx = 3000
        if case.endswith('nx1536') or case.endswith('nx3072') or case.endswith('nx8640'):
            nx = int(case[-4:])
        else:
            nx = 4320
        x = np.arange(dx/2, nx*dx, dx)

        svals = -S-(dS/2)*np.cos(2*np.pi*x/(dx*nx))  # S on original grid
        s96 = xr.DataArray(svals, coords={'x': x}).coarsen(
            x=32).mean()  # coarsen
        # shift x coordinate to match MSE chunked output
        s96['x'] = s96.x - s96.x[0]
        tmp['S'] = s96
    else:
        tmp['S'] = (('x'), np.full(tmp.x.values.shape, np.nan))
    tmp['S'].attrs = {'long_name': 'ocean heat source', 'units': 'W/m2'}

    # Mean SST for the simulation over the averaging period (important for slab ocean runs only)
    tmp['sst'] = (('sst'), [np.round(tmp.SST.mean().values, 1)])

    # Harmonize fill values before saving
    for var in tmp.variables:
        if hasattr(tmp[var], 'encoding'):
            if '_FillValue' in tmp[var].encoding and 'missing_value' in tmp[var].encoding:
                # Keep _FillValue and remove missing_value
                tmp[var].encoding.pop('missing_value', None)

    # Save
    fname = case + f'_d{str(int(t0))}-{str(int(t1))}.nc'
    fpath = os.path.join(DIR_MEAN_MSE, fname)
    if os.path.exists(fpath):
        os.remove(fpath)
    tmp.to_netcdf(fpath)

    # Clean up
    dd.close()
    tmp.close()
    # stat.close()
    print(fname)


def combine_time_averaged_output(fnames, new_file_name):
    """
    Combine the time-averaged output files for a list of cases.
    Saves the combined output to a netCDF file in DIR_MEAN_MSE
    Arguments:
        fnames: list of file names to combine
        new_file_name: name of the output file
    """

    # Load the datasets
    try:
        ds = xr.open_mfdataset(fnames).chunk({'sst': -1})
    except:
        dsets = []
        for f in fnames:
            dsets.append(xr.open_mfdataset(f))
        ds = xr.concat(dsets, dim='sst')

    # Convert all boolean variables to integers
    for var in ds.data_vars:
        if np.issubdtype(ds[var].dtype, np.bool_):
            ds[var] = ds[var].astype(int)

    if 'CO2' in new_file_name:
        co2_factors = []
        for file in fnames:
            if 'CO2' in file:
                # Extract the CO2 value from the file name
                factor_dict = {'0p7': 0.707, '0p8': 0.8, '0p9': 0.896, '1p1': 1.125,
                               '1p4': 1.414, '1p6': 1.63, '1p8': 1.772, '2': 2.0}
                co2_match = re.search(r'_(\d+\.?\d*|(?:\d+p\d*))xCO2', file)
                if co2_match:
                    co2_fac = factor_dict.get(co2_match.group(1))
                    co2_factors.append(co2_fac)
            else:
                co2_factors.append(1.0)
        ds['co2'] = (('sst'), 348*np.array(co2_factors))

    # Save the combined dataset
    save_name = os.path.join(
        DIR_MEAN_MSE_COMBINED, new_file_name)
    if os.path.exists(save_name):
        os.remove(save_name)
    ds.to_netcdf(save_name)
    print('Created new file: {}'.format(new_file_name))


if __name__ == "__main__":
    # Use 'spawn' for compatibility in cluster environments
    multiprocessing.set_start_method('spawn')

    if ncpu > 1:
        with multiprocessing.Pool(processes=ncpu) as pool:
            pool.map(combine_MSE_output, cases)

        create_time_averaged_output_partial = partial(
            create_time_averaged_output, avg_period=avg_period, tstart=tstart, tend=tend)

        with multiprocessing.Pool(processes=ncpu) as pool:
            pool.map(create_time_averaged_output_partial, cases)

    else:
        #for case in cases:
        #    combine_MSE_output(case)
        for case in cases:
            create_time_averaged_output(
                case, avg_period=avg_period, tstart=tstart, tend=tend)
        # for case in cases:
        #    create_pcoord_output(case, avg_period=avg_period, tstart = tstart, tend = tend
