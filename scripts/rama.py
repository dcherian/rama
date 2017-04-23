import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime as dt
import cmocean as cmo
import seawater as sw
import netCDF4 as nc
import sys
sys.path.append('/home/deepak/python')
import dcpy.oceans as do
import dcpy.plots
import dcpy.util


def Initialize(name, fname):

    # setup a mooring dictionary
    rama = dict([])
    rama['name'] = name
    rama['sal'] = dict([])
    rama['temp'] = dict([])
    rama['dens'] = dict([])
    rama['cond'] = dict([])
    rama['sal-hr'] = dict([])
    rama['temp-hr'] = dict([])
    rama['dens-hr'] = dict([])
    rama['N2fit'] = dict()
    rama['N2z'] = dict()

    rama = ReadPrelimData(rama, fname)

    return rama


def DefineDataTypes():
    """ Defines data types for reading ASCII data files. """

    sal = np.dtype([('date', dt.datetime),
                    ('sal', [('1', np.float32),
                             ('10', np.float32),
                             ('20', np.float32),
                             ('40', np.float32),
                             ('60', np.float32),
                             ('100', np.float32)]),
                    ('QQQQQQ', np.uint32),
                    ('SSSSSS', np.uint32)])

    cond = np.dtype([('date', dt.datetime),
                     ('cond', [('1', np.float32),
                               ('10', np.float32),
                               ('20', np.float32),
                               ('40', np.float32),
                               ('60', np.float32),
                               ('100', np.float32)]),
                     ('QQQQQQ', np.uint32),
                     ('SSSSSS', np.uint32)])

    temp = np.dtype([('date', dt.datetime),
                     ('temp', [('1', np.float32),
                               ('10', np.float32),
                               ('13', np.float32),
                               ('20', np.float32),
                               ('40', np.float32),
                               ('60', np.float32),
                               ('80', np.float32),
                               ('100', np.float32),
                               ('120', np.float32),
                               ('140', np.float32),
                               ('180', np.float32),
                               ('300', np.float32),
                               ('500', np.float32)]),
                     ('QQQQQQ', np.uint32),
                     ('SSSSSS', np.uint32)])

    dens = np.dtype([('date', dt.datetime),
                     ('dens', [('1', np.float32),
                               ('10', np.float32),
                               ('20', np.float32),
                               ('40', np.float32),
                               ('60', np.float32),
                               ('100', np.float32)]),
                     ('QQQQQQ', np.uint32),
                     ('SSSSSS', np.uint32)])

    return [sal, cond, temp, dens]


def RamaStitch(ra1, ra2):

    rama = dict()
    rama['name'] = ra1['name'] + ra2['name']
    rama['date'] = np.concatenate((ra1['date'], ra2['date']), axis=0)
    for pp in ['densarr', 'salarr', 'temparr', 'presarr']:
        rama[pp] = np.concatenate((ra1[pp], ra2[pp]), axis=1)

    return rama


def ReadPrelimData(rama, fname):
    """ Read data from ASCII file and process."""

    def Clean(value):
        """ Adds NaNs in place of missing values. """
        import numpy as np

        value = np.float32(value)

        if value > 100:
            value = np.nan

        return value

    def ProcessDate(datestr):
        """ Takes in string of form YYYYydayHHMM and returns
           python datetime object."""
        import datetime as dt

        year = int(datestr[0:4])
        yday = int(datestr[4:7])
        hour = int(datestr[7:9])
        mins = int(datestr[9:11])

        date = dt.datetime(year=year, month=1, day=1) \
                                 +  dt.timedelta(days=yday-1,
                                                 hours=hour,
                                                 minutes=mins)
        return date

    def ReadFile(rama, var, fname, dtype, converters):
        ''' Actually reads the text file. '''
        ramapre = np.loadtxt('../TAO_raw/' + var + fname + 'a.flg',
                             skiprows=5, dtype=dtype, converters=converters)
        rama[var + '-pre'] = ramapre[var]
        rama['date'] = ramapre['date']
        try:
            ramapost = np.loadtxt('../TAO_raw/postcal/sal' + fname + 'a.flg',
                                  skiprows=5, dtype=dtype,
                                  converters=converters)
            rama[var + '-post'] = ramapost[var]
        except FileNotFoundError:
            rama[var + '-post'] = rama[var + '-pre']

    [sal, cond, temp, dens] = DefineDataTypes()
    window_len = 13

    cnv = {0: ProcessDate}
    for jj in np.arange(1, 7):
        cnv[jj] = Clean

    ReadFile(rama, 'sal', fname, sal, cnv)
    ReadFile(rama, 'cond', fname, cond, cnv)
    ReadFile(rama, 'dens', fname, dens, cnv)

    for jj in np.arange(1, 14):
        cnv[jj] = Clean
    ReadFile(rama, 'temp', fname, temp, cnv)

    rama['hr-time'] = rama['date'][0::window_len/2]

    # ########### Interpolate and convert to dictionaries
    Ntime = len(rama['date'])

    weight_pre = np.arange(Ntime-1, -1, -1) / (Ntime-1)
    weight_post = np.arange(0, Ntime) / (Ntime-1)

    for depth in rama['sal-pre'].dtype.names:
        rama['dens-pre'][depth] = rama['dens-pre'][depth] + 1000
        rama['dens-post'][depth] = rama['dens-post'][depth] + 1000
        rama['temp'][depth] = smooth(rama['temp-pre'][depth], window_len)

        # pre to post-cal interpolation
        rama['cond'][depth] = smooth(weight_pre * rama['cond-pre'][depth]
                                     + weight_post * rama['cond-post'][depth],
                                     window_len)

        pres = sw.eos80.pres(float(depth), 12)
        rama['sal'][depth] = sw.eos80.salt(rama['cond'][depth]
                                           / sw.constants.c3515,
                                           rama['temp'][depth],
                                           pres)

        rama['dens'][depth] = sw.pden(rama['sal'][depth],
                                      rama['temp'][depth],
                                      pres)

        # filter hourly
        rama['temp-hr'][depth] = smooth(rama['temp'][depth],
                                        window_len)[0::window_len/2]
        rama['sal-hr'][depth] = smooth(rama['sal'][depth],
                                       window_len)[0::window_len/2]
        rama['dens-hr'][depth] = smooth(rama['dens'][depth],
                                        window_len)[0::window_len/2]

    rama['hr-time'] = rama['date'][0::window_len/2]

    ReadDailyData(rama)

    return rama


def SaveRama(rama, proc=''):
    ''' This saves a (depth, time) matrix of temp, sal, pres to
    RamaPrelimProcessed/rama['name'].mat '''

    from scipy.io import savemat

    def datetime2matlabdn(dt):
        import datetime as date
        mdn = dt + date.timedelta(days=366)
        frac = (dt - date.datetime(dt.year, dt.month, dt.day,
                                   0, 0, 0)).seconds \
                                / (24.0 * 60.0 * 60.0)
        return mdn.toordinal() + frac

    MakeArrays(rama, proc)

    if proc is '':
        datevec = rama['date']
    else:
        if proc[0] is '-':
            proc = proc[1:]

        datevec = rama[proc + '-time']

    datenum = np.array([datetime2matlabdn(dd) for dd in datevec])
    mdict = {'time': datenum,
             'sal': rama['salarr'],
             'temp': rama['temparr'],
             'depth': rama['presarr'][:, 0],
             'N2': rama['N2fit'],
             'N2z': rama['N2z']}

    savemat('../RamaPrelimProcessed/' + rama['name'],
            mdict, do_compression=True)


# read netCDF data
def ReadDailyData(rama, salfilename='../s12n90e_dy.cdf',
                  tempfilename='../t12n90e_dy.cdf'):
    import netCDF4 as nc

    salfile = nc.Dataset(salfilename)
    tempfile = nc.Dataset(tempfilename)

    # t0 = np.datetime64(salfile['time'].units[14:])
    t0 = dt.datetime.strptime(salfile['time'].units[11:],
                              '%Y-%m-%d %H:%M:%S')
    timevec = np.array([t0 + dt.timedelta(days=tt.astype('float'))
                        for tt in salfile['time'][0:]])

    indstart = np.argmin(np.abs(timevec - rama['date'][0]))
    indstop = np.argmin(np.abs(timevec - rama['date'][-1]))

    tindex = [np.where(tempfile['depth'][:] == zz)[0][0]
              for zz in salfile['depth'][:]]
    temp_matrix = tempfile['T_20'][indstart:indstop+1].squeeze()
    temp_matrix[temp_matrix > 40] = np.nan
    sal_matrix = salfile['S_41'][indstart:indstop+1].squeeze()
    sal_matrix[sal_matrix > 40] = np.nan

    dens_matrix = sw.pden(sal_matrix, temp_matrix[:, tindex],
                          salfile['depth'][:])

    # save processed salinity product
    rama['sal-dy'] = dict([])
    rama['temp-dy'] = dict([])
    rama['dens-dy'] = dict([])
    rama['dy-time'] = timevec[indstart:indstop+1]

    for index, zz in enumerate(np.int32(salfile['depth'][:])):
        rama['sal-dy'][str(zz)] = sal_matrix[:, index]
        rama['temp-dy'][str(zz)] = temp_matrix[:, tindex[index]]
        rama['dens-dy'][str(zz)] = dens_matrix[:, index]


def MakeArrays(rama, proc=''):

    rama['salarr'] = np.array([rama['sal' + proc]['1'],
                               rama['sal' + proc]['10'],
                               rama['sal' + proc]['20'],
                               rama['sal' + proc]['40'],
                               rama['sal' + proc]['60'],
                               rama['sal' + proc]['100']])

    rama['temparr'] = np.array([rama['temp']['1'],
                                rama['temp']['10'],
                                rama['temp']['20'],
                                rama['temp']['40'],
                                rama['temp']['60'],
                                rama['temp']['100']])

    rama['densarr'] = np.array([rama['dens' + proc]['1'],
                                rama['dens' + proc]['10'],
                                rama['dens' + proc]['20'],
                                rama['dens' + proc]['40'],
                                rama['dens' + proc]['60'],
                                rama['dens' + proc]['100']])

    rama['presarr'] = sw.pres(np.array([1*np.ones(rama['salarr'][0, :].shape),
                                        10*np.ones(rama['salarr'][0, :].shape),
                                        20*np.ones(rama['salarr'][0, :].shape),
                                        40*np.ones(rama['salarr'][0, :].shape),
                                        60*np.ones(rama['salarr'][0, :].shape),
                                        100*np.ones(rama['salarr'][0, :].shape)]),
                              12)
    return rama


def Compare10mDyDiff(rama, var, proc='', filt=False, window_len=13):
    ''' Compares 10m and daily differences of quantities '''
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    monthsFmt = mpl.dates.DateFormatter("%d-%m")

    if var is 'sal':
        label = 'S'

    if var is 'temp':
        label = 'T'

    if var is 'dens':
        label = 'ρ'

    if proc is not '' and proc[0] is not '-':
        proc = '-' + proc

    if filt is False:
        window_len = 1

    limy = [-0.2, 0.4]
    depths = list(rama[var].keys())
    for index, [d1, d2] in enumerate(zip(depths[0:-3], depths[1:-2])):
        hax = plt.subplot(3, 1, index+1)
        dens1 = smooth(rama[var + proc][d1], window_len=window_len)
        dens2 = smooth(rama[var + proc][d2], window_len=window_len)
        plt.plot(rama['date'][0::window_len/2],
                 dens2[0::window_len/2]-dens1[0::window_len/2], linewidth=1)
        plt.plot(rama['dy-time'],
                 rama[var + '-dy'][d2] - rama[var + '-dy'][d1], linewidth=1)
        plt.axhline(0, color='k')
        if var is 'sal':
            plt.axhline(0.06, color='gray')
            plt.axhline(-0.06, color='gray')

        plt.ylabel('Δ' + label + ' ' + d2 + 'm-' + d1 + 'm')
        plt.ylim(limy)
        hax.xaxis.set_major_formatter(monthsFmt)

    plt.gcf().suptitle(proc)
    plt.show()


def Compare10mDy(rama, var, proc=''):
    ''' Plots 10min and daily timeseries of var'''
    if var is 'sal':
        label = 'S'

    if var is 'temp':
        label = 'T'

    if var is 'dens':
        label = 'ρ'

    if proc is not '' and proc[0] is not '-':
        proc = '-' + proc

    for index, zz in enumerate(['1', '10', '20', '40']):
        plt.subplot(4, 1, index+1)
        datenum = mpl.dates.date2num(rama['date'])
        plt.plot(datenum, rama[var + proc][zz], linewidth=1)
        plt.ylabel(label + ' ' + zz + 'm')
        plt.plot(rama['dy-time'], rama[var + '-dy'][zz], linewidth=1)

    plt.gcf().suptitle(proc)
    plt.show()


def PcolorAll(rama, ylim=None):
    ''' Pcolor T, S, ρ '''
    try:
        MakeArrays(rama)
    except:
        pass

    ax1 = plt.subplot(311)
    PcolorProperty(rama, 'temp', ylim)
    ax2 = plt.subplot(312, sharex=ax1)
    PcolorProperty(rama, 'sal', ylim)
    ax3 = plt.subplot(313, sharex=ax1)
    PcolorProperty(rama, 'dens', ylim)
    plt.tight_layout()
    plt.show()


def PcolorProperty(rama, varname, ylim=None):
    import matplotlib as mpl

    if varname is 'sal':
        # color = cmo.cm.haline_r
        color = plt.cm.OrRd
        clim = [31.5, 35]

    if varname is 'temp':
        color = cmo.cm.thermal
        clim = [25, 31]

    if varname is 'dens':
        color = cmo.cm.dense
        clim = [1019, 1023]

    sz = rama[varname + 'arr'].shape
    datex = np.tile(mpl.dates.date2num(rama['date']), (sz[0], 1))
    plt.contourf(datex, -rama['presarr'],
                 np.ma.masked_array(rama[varname + 'arr'],
                                    np.isnan(rama[varname + 'arr'])),
                 20, cmap=color)
    plt.colorbar()
    xfmt = mpl.dates.DateFormatter('%Y-%m')
    plt.gca().xaxis.set_major_formatter(xfmt)
    plt.gcf().autofmt_xdate()
    plt.clim(clim)
    plt.title(rama['name'] + ' | ' + varname)
    plt.axhline(-15, color='w', linewidth=1)
    plt.axhline(-30, color='w', linewidth=1)
    if ylim is not None:
        plt.ylim(ylim)


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that # TODO: ransient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
        flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

     see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning',"
                         + "'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[(window_len/2-1):-(window_len/2+1)]


def ArgoPlot(rama, date, floatname='5904313'):
    """ Plot Argo profile against RAMA profile. """
    argo = nc.Dataset('../argo/' + floatname + '/'
                      + floatname + '_prof.nc', 'r')

    adate = argo['JULD'][:] \
            + mpl.dates.date2num(
                mpl.dates.datetime.datetime(1950, 1, 1))

    profile = np.argmin(np.abs(adate
                               - mpl.dates.date2num(date)))
    time = mpl.dates.num2date(adate[profile])
    plt.title(str(time))
    plt.plot(argo['PSAL_ADJUSTED'][profile, :],
             -argo['PRES_ADJUSTED'][profile, :])
    plt.ylim([-120, 0])

    ind = np.argmin(np.abs(mpl.dates.date2num(rama['date']) - adate[profile]))
    MakeArrays(rama, '-post')
    plt.plot(rama['salarr'][:, ind], -rama['presarr'][:, ind])
    MakeArrays(rama, '-pre')
    plt.plot(rama['salarr'][:, ind], -rama['presarr'][:, ind])
    MakeArrays(rama, '')
    plt.plot(rama['salarr'][:, ind], -rama['presarr'][:, ind])
    plt.legend(('argo', 'post', 'pre', 'full'))
    plt.show()


def ScatterTS(rama, depth, tlim=None, woa=None):

    T = rama['temp'][depth][:]
    S = rama['sal'][depth][:]
    t = mpl.dates.date2num(rama['date'][:])

    if tlim is not None:
        trange = np.where(np.logical_and(t >= tlim[0], t <= tlim[1]))[::6]
    else:
        trange = np.arange(0, len(S))

    colormap = cmo.cm.matter

    plt.scatter(S[trange], T[trange], s=30, c=t[trange],
                alpha=0.65, linewidth=0.15, edgecolor='gray',
                cmap=colormap)
    # fmt = mpl.dates.DateFormatter('%Y-%m')
    # plt.colorbar(format=fmt)
    if tlim is not None:
        plt.clim(tlim)
        plt.title(rama['name'] + ' | ' + depth + ' m')
        plt.xlabel('S')
        plt.ylabel('T')

    if woa is not None:
        from scipy.stats import mode
        # figure out year
        year = mode(np.array([dd.year
                              for dd in mpl.dates.num2date(t[trange])]))[0]
        woatime = [np.float32(mpl.dates.date2num(dt.datetime(year, ii, 1)))
                   for ii in np.arange(1, 13)]
        index = np.where(woa['depth'] == int(depth))
        plt.scatter(woa['S'][:, index], woa['T'][:, index],
                    s=4*120, c=np.array(woatime),
                    alpha=0.5, linewidth=0.15, edgecolor='gray',
                    cmap=colormap, zorder=-10)


def CalcGradients(rama):
    dSdz = -np.diff(rama['salarr'], axis=0)/np.diff(rama['presarr'], axis=0)
    dTdz = -np.diff(rama['temparr'], axis=0)/np.diff(rama['presarr'], axis=0)

    N2, _, p_ave = sw.bfrq(rama['salarr'],
                           rama['temparr'],
                           rama['presarr'], 12)
    rama['N2'] = N2
    return (dSdz, dTdz, N2, p_ave)


def N2fit(ρ, depth, depth0=None, curve='tanh', doplot=False,
          interp=False, tt=None):
    ''' Determine N² by fitting curve to ρ profile.

    Input:
        ρ = fn.(depth, time)
        depth = depth vector
        depth0 = Depth at which you want N²
                 if None, return all depths
        curve = tanh (default) / spline
        doplot = if True, plot fit against data
        interp = if True, interpolate to denser grid before fitting

    Output:
        N2 = fn.(time)
        N2z = depth grid for N2
    '''

    import numpy as np

    if depth0 is None:
        depth0 = depth

    N2z = np.asarray(np.float64(depth0))
    depth = np.float64(depth)

    if tt is None:
        ntime = ρ.shape[1]
    else:
        ntime = 1

    if ntime == 1:
        N2 = CalN2(ρ[:, tt], depth, N2z, tt=tt, curve=curve, doplot=doplot)
        if doplot:
            N2line = 9.81/1025*(ρ[2, tt]-ρ[1, tt])/(depth[2] - depth[1])
            import matplotlib.pyplot as plt
            plt.title('N2fit = ' + "{:1.3e}".format(N2)
                      + ' | N2line = ' + "{:1.3e}".format(N2line))

    else:
        from joblib import Parallel, delayed

        N2 = Parallel(n_jobs=-1)(delayed(CalN2)(ρ[:, t], depth, N2z, t, curve)
                                 for t in np.arange(ntime))

    return [np.array(N2), N2z]


def CalN2(ρ, depth, N2z, tt, curve='tanh', interp=False, doplot=False):
    ''' Used for parallel call from N2Tanh below. '''
    from dcpy.fits import fit
    import numpy as np

    mask = np.isfinite(ρ)
    if np.sum(mask) < 3:
        # less than 3 valid points
        return np.nan

    mw = 2  # max weight
    weights = np.array([1, 1e-1, mw, 1, 1e-1, 1e-4])

    # accounts for summer peak in N²
    if tt < 22430 and tt > 18700:
        weights[1] = mw+2
        weights[2] = mw+2
        weights[-2] = 1
    else:
        if tt > 40000:
            weights[1] = mw
            weights[-1] = 1

    num_try = 0
    Z = 1e4
    Zthresh = 1000

    if mask[0] is False and weights[1] < 1:
        # surface sensor has failed. need to use 10m sensor
        weights[1] = mw

    if curve == 'spline':
        spl = fit(curve, depth, ρ, weights=weights, doplot=doplot, k=2)
        dρdz = spl.derivative(1)(N2z)

    else:
        while np.abs(Z) > Zthresh and num_try <= 7:
            weights[weights < 1] *= 10**(num_try)
            num_try += 1
            # if interp:
            #     ddense = np.linspace(depth.min(), depth.max(), 10)
            #     r = np.interp(ddense, depth[mask], ρ[mask])
            #     [y0, Z, z0, y1] = fit(curve, ddense, r,
            #                           weights, doplot=doplot, maxfev=100000)
            # else:
            try:
                [y0, Z, z0, y1] = fit(curve, depth[mask], ρ[mask],
                                      weights[mask],
                                      doplot=doplot, maxfev=100000)
            except RuntimeError:
                # fit did not work
                [y0, Z, z0] = [np.nan, np.nan, np.nan]

        if np.abs(Z) > Zthresh:
            Z = np.nan

        dρdz = y0/Z * (1 - np.tanh((N2z-z0)/Z)**2)

    # if provided with density return N²
    # else just return gradient (probably provided with salinity)
    if np.all(ρ > 200):
        return 9.81/1025 * dρdz
    else:
        return dρdz


def TabulateNegativeN2(p_ave, N2, dSdz, dTdz):
    ''' Percentage of valid data that yields N² < 0 '''
    table = [list(np.round(p_ave[:,0])),
             [np.round(len(n[n<0])/len(n)*100) for n in # % N² < 0
              [N2[i,~np.isnan(N2[i,:])] for i in range(N2.shape[0])]],
             [np.round(len(s[s>0])/len(s)*100) for s in # % dS/dz > 0
              [dSdz[i,~np.isnan(dSdz[i,:])] for i in range(dSdz.shape[0])]],
             [np.round(len(s[s<0])/len(s)*100) for s in # % dT/dz > 0
              [dTdz[i,~np.isnan(dTdz[i,:])] for i in range(dTdz.shape[0])]]]

    table[0].insert(0, 'Depth (m)')
    table[1].insert(0, '% N² < 0')
    table[2].insert(0, '% dS/dz > 0')
    table[3].insert(0, '% dT/dz < 0')

    return table
