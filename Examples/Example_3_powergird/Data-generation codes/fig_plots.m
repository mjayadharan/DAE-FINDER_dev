%% Plotting networks

clear all; close all; clc;
addpath([pwd,'/MATPOWER_power_grids/'])
knob.gamma = 1;
% matcase = 'case_4bus2gen'
matcase = 'case_39bus10gen_newengland'

% MATPOWER case
mpc = feval(matcase);
mpc.ref_freq = 60;

% Power grid with no losses (gamma = 0)
if knob.gamma == 0
    mpc.branch(:,3) = 0;
    disp('Warning: gamma = 0')
end

% Structure-preserving model
[p, details] = SP_model(mpc);
ng = details.ngi; 
nl = details.nl;
N = 2*ng+nl;
n = 3*ng+nl;
K = p.K .* (ones(N,N)-eye(N,N));

figure(1); 
Kgraph = (K+K')/2;
% h = plot(graph(Kgraph),'Layout','subspace','Dimension',200)
h = plot(graph(Kgraph),'Layout','force','Iterations',40)
highlight(h,[1:1:ng],'NodeColor',0.1*[1 1 1])%[0.682, 0.560, 0.337])%
highlight(h,[ng+1:1:N],'NodeColor',0.6*[1 1 1])
highlight(h,[2*ng+1:1:N],'Marker','s')
highlight(h,[1:1:ng],'MarkerSize',8)
highlight(h,[ng+1:1:2*ng],'MarkerSize',7)
highlight(h,[2*ng+1:1:N],'MarkerSize',8)
h.EdgeColor = 0.5*[1 1 1]
h.LineWidth = 2
h.NodeLabel = {};

%% Plotting dynamics
clear all; close all; clc;
T = readtable('case_39bus10gen_newengland_timeseries_half.csv');

step = 10;
t0 = 1;
tf = 200010;

N = 39+10;
ng = 10;
nl = 29;
n = 59;

time = table2array(T(t0:step:tf,1));
phase = table2array(T(t0:step:tf,2:N+1));
freq = table2array(T(t0:step:tf,N+2:N+ng+1));

figure(3);
subplot(3,1,1); plot(time,phase');

subplot(3,1,2); plot(time,phase');
xlim([2 4])


T = readtable('case_39bus10gen_newengland_timeseries_onetenth.csv');
time = table2array(T(t0:step:tf,1));
phase = table2array(T(t0:step:tf,2:N+1));
freq = table2array(T(t0:step:tf,N+2:N+ng+1));

subplot(3,1,3); plot(time,phase');
xlim([2 4])
xlabel('time');


fontsize(16,"points")
print(gcf,'-vector','-dsvg','timeseries.svg') % svg
