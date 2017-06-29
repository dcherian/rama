% test out nmatlab spectra code before attempting the same exercise in python

load ~/rama/RAMA13/data/527/proc/combined/2017-04-20.mat

time = Turb.chi_mm1.time;
dt = round(diff(time(1:2))*86400); % in seconds
L = round(3600/dt); % averaging time period
Lb2 = floor(L/2);
N = (86400/L/dt); % number of points / day

chi = nanmean([Turb.chi_mm1.chi; Turb.chi_mm2.chi], 1);
chim = movmean(chi, L, 'omitnan');
timem = movmean(time, L, 'omitnan');
chim = chim(Lb2:L:end);
timem = timem(Lb2:L:end);
%% spectra by multiple subsets

t0 = find_approx(timem, datenum(2014,8,1));
t1 = find_approx(timem, datenum(2014,8,31));
%MaxSubsetLength = 30*86400; % 1 month
timesub = timem(t0:t1);
chisub = chim(t0:t1);

dts = (timesub(2)-timesub(1))*86400;
ndays = 1;
figure;
hax = gca();
[S, f] = GappySpectrum(chisub, [], 5);
loglog(f/dts, S)

%%
T = nanmean([Turb.chi_mm1.T; Turb.chi_mm2.T], 1);
T = T(1:515140);
time = time(1:515140);

Tm = movmean(T, L, 'omitnan');
timem = movmean(time, L, 'omitnan');
Tm = Tm(Lb2:L:end);
timem = timem(Lb2:L:end);

%% just temp spectra
var = T;
fs = 1./(60/86400); % cpd
ndays = 120;
nbands = 12;
segment_len = ndays * fs;
% [S, f] = spectrum_band_avg(var, 1./fs, nbands, 'no taper', 1);
[S, f] = gappy_psd(var, segment_len, fs, 12);
% loglog(f, S); hold on;
loglog(smooth(f, nbands), smooth(S, nbands));

% time periods (s)
Tdaily = 86400;
Tf0 = 2*pi/sw_f(12);
TM2 = (12.42*3600);
TN = 2*pi/(1e-2);
linex(Tdaily./[Tdaily Tf0 TM2 TN])
linex([1./ndays fs/2])
xlabel('Frequency (cpd)')
ylabel('psd')
% xlim([1./(length(var)*60/86400) 1/(60*2/86400)] .* [0.8 1.2])
% linex([1./(length(var)*60/86400) 1/(60*2/86400) 1/365])

%%
ndays = 5;
figure; maximize;
spectrogram(T, hann(ndays*86400/dt+1), 0, [], 86400/dt,...
            'onesided', 'yaxis');
hold on;
hax = gca;
hax.YScale = 'log';
hax.Children(1).XData = avg1(time(1:ndays*86400/dt:end));
% hax.Children(1).ZData = log10(hax.Children(1).ZData);
xlim([min(time) max(time)])
uistack(hax.Children(1), 'bottom');
datetick('keeplimits');
xlabel('2014');
 ylabel('Frequency (cpd)')

 N2 = Turb.chi_mm1.N2(1:length(time));
 N2(N2<0) = NaN;
 TN = 2*pi./sqrt(N2);
 TN(N2 < 1e-7) = NaN;
 plot(time, movmean(86400./TN, 86400/dt, 'omitnan'), ...
      'Color', [1 1 1]*0.6, 'LineWidth', 1);

 hl = liney(Tdaily./[Tdaily Tf0 TM2], ...
            {'daily'; 'inertial'; 'M2'});
 uistack(hl, 'top');
 liney(Tdaily/Tf0, 'inertial');
 title('PSD(temp) + N2')
 set(gca, 'Color', 'none');
export_fig -transparent images/526-temp-spectrogram.png
%%
figure; maximize;
for kk=1:2
    for ii=2:5
        if kk == 1
            var = Tm;
        else
            var = chim;
        end

        [S, f] = gappy_psd(var, ii*30*N, N, 2);

        jj = ii-2+1;
        hax(kk, jj) = subplot(4, 2, sub2ind([2 4], kk, jj));
        loglog(f, S); hold on;
        title([num2str(ii*30) ' day segments']);
        beautify([13 14 16]*1.2); grid on;
        set(hax(kk, jj), 'Color', 'None');
        xlim([5e-3 1]);
        if kk == 2
            ylim([1e-12 1.3e-11]);
        else
            ylim([1e-4 10])
        end
        linex(1./[2.5 3], {'2.5'; '3'});
    end
    linkaxes(hax(kk,:), 'xy');
    xlabel('Frequency(cpd)');
end
linkaxes(hax, 'x');

% export_fig -r150 -transparent images/526-chi-proto-spectra.png

%% let's try sally's
load ~/rama/RAMA13/processed/chi_analysis_bkgrnd_Feb5/deglitched/mean_chi_526_mindTdz1e-3.mat

time = avgchi.time;
chi = nanmean([avgchi.chi1; avgchi.chi2], 1);
T = nanmean([avgchi.T1; avgchi.T2], 1);

var = chi;

[gapstart, gapend] = FindGaps(var);
clear range
if gapstart ~= gapend
    if gapstart(1) == 1, range(1) = gapend(1)+1; end
    if gapend(end) == length(var), range(2) = gapstart(end)-1; end
    range = range(1):range(2);
else
    range = 1:length(var);
end

var2 = fill_gap(var(range), 'linear', 1e10);
gamma = 3; beta = 2;
fs = morsespace(gamma,beta,{0.05,pi},pi/1000,4);
[w] = wavetrans(var2', {gamma,beta,fs,'bandpass'});
figure;
h = wavespecplot(time(range), vfilt(var2', 24*6), ...
                        1./fs, w);
if nanmin(var) < 0.1, h(1).YScale = 'log'; end
linkaxes(h, 'x')
datetick;
colormap(cbrewer('seq', 'Reds', 11));
colorbar('off')
