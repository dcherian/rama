import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime as dt
import cmocean as cmo
import seawater as sw
from copy import copy
import netCDF4 as nc

def Initialize(name, fname):

    # setup a mooring dictionary
    rama = dict([])
    rama['name'] = name
    rama['sal']  = dict([])
    rama['temp'] = dict([])
    rama['dens'] = dict([])
    rama['sal-hr'] = dict([])
    rama['temp-hr'] = dict([])
    rama['dens-hr'] = dict([])

def ReadPrelimData(rama):
    def CleanSalinity(salinity):
        """ Adds NaNs in place of missing values. """
        import numpy as np

        salinity = np.float32(salinity)

        if salinity > 39:
            salinity = np.nan

            return salinity

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

    ############# salinity
    sal = np.dtype([('date', dt.datetime),
		    ('sal', [('1', np.float32),
                             ('10', np.float32),
                             ('20', np.float32),
                             ('40', np.float32),
                             ('60', np.float32),
                             ('100', np.float32)]),
		        ('QQQQQQ', np.uint32),
		        ('SSSSSS', np.uint32)])

    cnv = {0:ProcessDate}
    for jj in np.arange(1,7):
        cnv[jj] = CleanSalinity;

    ramapre = np.loadtxt('../TAO_raw/sal' + fname + 'a.flg',
                         skiprows=5, dtype=sal, converters=cnv)
    rama['sal-pre'] = ramapre['sal']
    ramapost = np.loadtxt('../TAO_raw/postcal/sal' + fname + 'a.flg',
                          skiprows=5, dtype=sal, converters=cnv)
    rama['sal-post'] = ramapost['sal']

    rama['date'] = ramapre['date']
    rama['hr-time'] = rama['date'][0::window_len/2]

    ############## density
    dens = np.dtype([('date', dt.datetime),
		     ('dens', [('1', np.float32),
                               ('10', np.float32),
                               ('20', np.float32),
                               ('40', np.float32),
                               ('60', np.float32),
                               ('100', np.float32)]),
		     ('QQQQQQ', np.uint32),
		     ('SSSSSS', np.uint32)])

    ramapre = np.loadtxt('../TAO_raw/dens' + fname + 'a.flg',
                         skiprows=5, dtype=dens, converters=cnv)
    rama['dens-pre'] = ramapre['dens']

    ramapost = np.loadtxt('../TAO_raw/postcal/dens' + fname + 'a.flg',
                          skiprows=5, dtype=dens, converters=cnv)
    rama['dens-post'] = ramapost['dens']

    ############# temperature
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

    cnv = {0:ProcessDate}
    for jj in np.arange(1,14):
        cnv[jj] = CleanSalinity;

    ramapre = np.loadtxt('../TAO_raw/temp' + fname + 'a.flg',
                         skiprows=5, dtype=temp, converters=cnv)

    ############ Interpolate and convert to dictionaries
    Ntime = len(ramapre['date'])

    weight_pre = np.arange(Ntime-1,-1,-1)/(Ntime-1)
    weight_post = np.arange(0,Ntime)/(Ntime-1)

    window_len = 13
    for depth in rama['sal-pre'].dtype.names:
        rama['dens-pre'][depth] = rama['dens-pre'][depth] + 1000
        rama['dens-post'][depth] = rama['dens-post'][depth] + 1000
        rama['temp'][depth] = ramapre['temp'][depth]

        # pre to post-cal interpolation
        rama['sal'][depth] = weight_pre * rama['sal-pre'][depth] \
                             + weight_post * rama['sal-post'][depth]
        rama['dens'][depth] = weight_pre * rama['dens-pre'][depth] \
                              + weight_post * rama['dens-post'][depth]

        # filter hourly
        rama['temp-hr'][depth] = smooth(rama['temp'][depth],
                                        window_len)[0::window_len/2]
        rama['sal-hr'][depth] = smooth(rama['sal'][depth],
                                       window_len)[0::window_len/2]
        rama['dens-hr'][depth] = smooth(rama['dens'][depth],
                                        window_len)[0::window_len/2]

    ################## read netCDF data (daily frequency)
    salfilename='../s12n90e_dy.cdf'
    tempfilename='../t12n90e_dy.cdf'

    salfile = nc.Dataset(salfilename)
    tempfile = nc.Dataset(tempfilename)

    t0 = dt.datetime.strptime(salfile['time'].units[11:],
			      '%Y-%m-%d %H:%M:%S')
    timevec = np.array([t0 + dt.timedelta(days=tt.astype('float')) \
                                for tt in salfile['time'][0:]])

    ind107start = np.argmin(np.abs(timevec - rama['date'][0]))
    ind107stop = np.argmin(np.abs(timevec - rama['date'][-1]))

    tindex = [np.where(tempfile['depth'][:] == zz)[0][0]
                      for zz in salfile['depth'][:]]
    temp_matrix = tempfile['T_20'][ind107start:ind107stop+1].squeeze()
    temp_matrix[temp_matrix > 40] = np.nan
    sal_matrix = salfile['S_41'][ind107start:ind107stop+1].squeeze()
    sal_matrix[sal_matrix > 40] = np.nan

    dens_matrix = sw.pden(sal_matrix,
                          temp_matrix[:,tindex],
                          salfile['depth'][:])

    rama['sal-dy'] = dict([])
    rama['temp-dy'] = dict([])
    rama['dens-dy'] = dict([])
    rama['dy-time'] = timevec[ind107start:ind107stop+1]
    for index, zz in enumerate(np.int32(salfile['depth'][:])):
        rama['sal-dy'][str(zz)] = sal_matrix[:,index]
        rama['temp-dy'][str(zz)] = temp_matrix[:,tindex[index]]
        rama['dens-dy'][str(zz)] = dens_matrix[:,index]

    return rama

def Compare10mDyDiff(rama, var, proc='', filt=False, window_len=13):
    ''' Compares 10m and daily differences of quantities '''
    import matplotlib as mpl
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

    depths = list(rama[var].keys())
    for index, [d1, d2] in enumerate(zip(depths[0:-3], depths[1:-2])):
        hax = plt.subplot(3,1,index+1)
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

    for index,zz in enumerate(['1', '10', '20', '40']):
        plt.subplot(4,1,index+1)
        datenum = mpl.dates.date2num(rama['date'])
        plt.plot(datenum, rama[var + proc][zz], linewidth=1)
        plt.ylabel(label + ' ' + zz + 'm')
        plt.plot(rama['dy-time'], rama[var + '-dy'][zz], linewidth=1)

    plt.gcf().suptitle(proc)
    plt.show()

def SaveRama(rama, proc=''):
    ''' This saves a (depth, time) matrix of temp, sal, pres to
    RamaPrelimProcessed/rama['name'].mat '''

    from scipy.io import savemat

    def datetime2matlabdn(dt):
        import datetime as date
        ord = dt.toordinal()
        mdn = dt + date.timedelta(days = 366)
        frac = (dt-date.datetime(dt.year,dt.month,dt.day,0,0,0)).seconds \
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
    mdict = {'time' : datenum,
	     'sal' : rama['salarr'],
	     'temp' : rama['temparr'],
	     'depth' : rama['presarr'][:,0]}

    savemat('RamaPrelimProcessed/' + rama['name'], mdict, do_compression=True)
