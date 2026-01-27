function [t,X] = odeRK(h_ode,T0,X0)
% implements the numerical integration algorithm Runge Kuta 4th order. The
% function works like 'ode45' from Matlab.
%   Inputs:
%       - h_ode: ode function handle
%           ex.: use '@rossler' for a function named 'rossler'
%       - T0: vector with intial and final integration time [t0 tf]
%           ex.: use [0 100] for a simulation from t=0 to t=100
%           ex.: use [0 0.0001 100] for t=0 to t=100, with step 0.0001
%       - X0: vector with intial values size(X0)=(1,n), n: system order
%           ex.: X0=[2 1 1] for a 3rd order system
% 
% MACSIN
% This code was was adapted from the LAA 31/12/16

% time vector
t0 = T0(1);     % initial time
h = T0(2);      % integration step
tf = T0(end);   % final time
t = t0:h:tf;    % time vector

% interesting variables
n = length(X0); % system order
N = length(t);  % # of points

% initial state
X = [X0; zeros(N-1,n)];

for k=2:length(t)
    x0 = X(k-1,:);
    
    % 1st call
    xd=feval(h_ode,t(k),x0)';
    savex0=x0;
    phi=xd;
    x0=savex0+0.5*h*xd;

    % 2nd call
    xd=feval(h_ode,t(k)+0.5*h,x0)';
    phi=phi+2*xd;
    x0=savex0+0.5*h*xd;

    % 3rd call
    xd=feval(h_ode,t(k)+0.5*h,x0)';
    phi=phi+2*xd;
    x0=savex0+h*xd;

    % 4th call
    xd=feval(h_ode,t(k)+h,x0)';
    X(k,:) = savex0+(phi+xd)*h/6;
    
end
