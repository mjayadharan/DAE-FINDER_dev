%% Generates data for the system identification of a power grid SP model.
clear all; close all; clc;
addpath([pwd,'/MATPOWER_power_grids/'])
knob.gamma = 1;
matcase = 'case_39bus10gen_newengland'

%% MATPOWER case
mpc = feval(matcase);
mpc.ref_freq = 60;

% Power grid with no losses (gamma = 0)
if knob.gamma == 0
    mpc.branch(:,3) = 0;
    disp('Warning: gamma = 0')
end

% Structure-preserving model
[p, details] = SP_model(mpc);

%% Kuramoto network
wref = p.omega_R; H = p.H; D = p.D; A = p.A; gamma = full(p.gamma);
K = p.K; phi = p.phi; 

% Algebraic parameters
Pi = details.Pi;
V = details.V;
Y_SP = details.Y_SP;

% Network size
ng = details.ngi; 
nl = details.nl;
N = 2*ng+nl;
n = 3*ng+nl;
K = K.*(ones(N,N)-eye(N,N));

% Admittance matrix
Y0 = details.Y0;
G = graph((Y0~=0) - diag(diag(Y0~=0)));

%% Continuous-time simulation parameters
t0 = 0.0; tf = 2.0; dt = 0.0001;
tspan = t0:dt:tf; T = length(tspan);

% Experimental setup
Nexp = 100;
pert_std = 0.1;

% System simulation
t = []; x = [];
for k = 1:Nexp
    if k == 1
        x0 = [phi; zeros(ng,1); phi; zeros(nl,1)];
    else
        x0 = x(:,end) + [pert_std*randn(ng,1); zeros(ng,1); pert_std*randn(ng,1); pert_std*randn(nl,1)];
        % x0 = x(:,end) + [zeros(ng,1); pert_std*randn(ng,1); zeros(ng,1); zeros(nl,1)]
    end
    [ttemp,xtemp] = odeRK(@(t,theta)kuramotoPG_SP(t,theta,N,ng,ng,nl,wref,K,H,A,D,gamma,zeros(n,1),0),[0 dt tf], x0');
    if k == 1
        t = ttemp;
        timepoint(k) = ttemp(1);
    else
        timepoint(k) = ttemp(1) + t(end);
        t = [t t(end)*ones(1,length(ttemp))+ttemp(1:end)];
    end
    x = [x xtemp(1:end,:)'];
end
t = t(1:10:end);
x = x(:,1:10:end);
phi = [x(1:ng,:); x(2*ng+1:2*ng+ng+nl,:)];

% Determines frequency for all oscillators
for k = 1:length(x)
    dxdt(:,k) = kuramotoPG_SP(t,x(:,k),N,ng,ng,nl,wref,K,H,A,D,gamma,zeros(n,1),0);
end
omega = [x(ng+1:2*ng,:); dxdt(2*ng+1:2*ng+ng+nl,:)];

%% Power dynamics
for k = 1:1:length(t)
    k
    for i = 1:N
        P(i,k) = 0;
        Q(i,k) = 0;
        for j = 1:N
            P(i,k) = P(i,k) + abs(V(i)*V(j)*Y_SP(i,j)) * sin( phi(i,k) - phi(j,k) + gamma(i,j) );
            Q(i,k) = Q(i,k) - abs(V(i)*V(j)*Y_SP(i,j)) * cos( phi(i,k) - phi(j,k) + gamma(i,j) );
        end
    end
end

%% Plots
figure(2)
subplot(141)
plot(t,phi)
title('Phase')
xlabel('t [s]'); ylabel('\theta_i [rad]')

subplot(142)
plot(t,omega)
title('Frequency')
xlabel('t [s]'); ylabel('\omega_i [rad/s]')

subplot(143)
plot(t,P)
title('Active power')
xlabel('t [s]'); ylabel('P_i')
hold on
plot(t,sum(P,1),'Color','black','LineWidth',3)

subplot(144)
plot(t,Q)
title('Reactive power')
xlabel('t [s]'); ylabel('Q_i')
hold on
plot(t,sum(P,1),'Color','black','LineWidth',3)


%% Saves data

% Time-series data
timeseries = [t' phi' omega' P' Q'];
headers0 = [{'time'}];

% Phases
generatorNodesHeaders = arrayfun(@(x) sprintf('PhaseGen%d', x), 1:ng, 'UniformOutput', false);
generatorTerminalsHeaders = arrayfun(@(x) sprintf('PhaseGenTerm%d', x), 1:ng, 'UniformOutput', false);
loadNodesHeaders = arrayfun(@(x) sprintf('PhaseLoad%d', x), 1:nl, 'UniformOutput', false);
headers1 = [generatorNodesHeaders, generatorTerminalsHeaders, loadNodesHeaders];

% Frequencies
generatorNodesHeaders = arrayfun(@(x) sprintf('FreqGen%d', x), 1:ng, 'UniformOutput', false);
generatorTerminalsHeaders = arrayfun(@(x) sprintf('FreqGenTerm%d', x), 1:ng, 'UniformOutput', false);
loadNodesHeaders = arrayfun(@(x) sprintf('FreqLoad%d', x), 1:nl, 'UniformOutput', false);
headers2 = [generatorNodesHeaders, generatorTerminalsHeaders, loadNodesHeaders];

% Power
generatorNodesHeaders = arrayfun(@(x) sprintf('ActivePowerGen%d', x), 1:ng, 'UniformOutput', false);
generatorTerminalsHeaders = arrayfun(@(x) sprintf('ActivePowerGenTerm%d', x), 1:ng, 'UniformOutput', false);
loadNodesHeaders = arrayfun(@(x) sprintf('ActivePowerLoad%d', x), 1:nl, 'UniformOutput', false);
headers3 = [generatorNodesHeaders, generatorTerminalsHeaders, loadNodesHeaders];

generatorNodesHeaders = arrayfun(@(x) sprintf('ReactivePowerGen%d', x), 1:ng, 'UniformOutput', false);
generatorTerminalsHeaders = arrayfun(@(x) sprintf('ReactivePowerGenTerm%d', x), 1:ng, 'UniformOutput', false);
loadNodesHeaders = arrayfun(@(x) sprintf('ReactivePowerLoad%d', x), 1:nl, 'UniformOutput', false);
headers4 = [generatorNodesHeaders, generatorTerminalsHeaders, loadNodesHeaders];

% Writes table
headers = [headers0 headers1 headers2 headers3 headers4];

T1 = array2table(timeseries, 'VariableNames', headers);
writetable(T1,[matcase,'_timeseries.csv']);

% Static data
staticdata = [V A wref*ones(N,1) D [H; zeros(ng+nl,1)]];
T2 = array2table(staticdata);
T2.Properties.VariableNames(1:5) = {'V','A','omega_R','D','H'}
writetable(T2,[matcase,'_staticparams.csv'])

% Timepoints
T3 = array2table(timepoint', 'VariableNames', {'Perturbation timepoints'});
writetable(T3,[matcase,'_timepoints.csv']);

% Saves table
csvwrite([matcase,'_Y.csv'],full(Y_SP))
csvwrite([matcase,'_K.csv'],full(K))
csvwrite([matcase,'_gamma.csv'],gamma)

