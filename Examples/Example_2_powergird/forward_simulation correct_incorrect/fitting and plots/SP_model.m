function [p, details] = SP_model(mpc, est_dyn)
%SP_model   Structure-preserving model for power-grid synchronization.
%   p = SP_model(mpc) runs a power flow calculation using MATPOWER and
%   computes the paramters of the structure-preserving (SP) model for the
%   dynamics of the power system given by struct mpc in the MATPOWER
%   format. The model is described in detail in Ref. [1]. The additional
%   field ref_freq of mpc can be used to specify the system reference
%   frequency in Hz (which defaults to 60 Hz if missing). The result will
%   be returned as a struct p, which will have the following fields:
%
%       omega_R: Angular reference frequency of the system in radian
%             H: Column vector of inertia constants H_i in seconds
%             D: Column vector of damping coefficients D_i in seconds
%             A: Column vector of constants A_i for the SP model
%             K: Coupling matrix K_{ij} for the SP model
%         gamma: Phase shif matrix gamma_{ij} for the SP model
%
%   The dynamic parameters for the generators can be defined in the matrix
%   mpc.gen_dyn. The columns of this matrix are:
%
%       1: Transient reactances x'_{d,i} in p.u.
%       2: Inertia constants H_i in seconds
%       3: (Combined) damping coefficients D_i in seconds
%
%   For load nodes, including generator terminal nodes, coefficients D_i
%   (in seconds) for the frequency dependence of real power consumption can
%   be defined in the vector mpc.load_dyn.  Each component corresponds to a
%   generator terminal node or a load node in the order they appear in
%   mpc.bus.
%
%   These parameters should be given on the system base MVA specified by
%   mpc.baseMVA. Any missing parameters will be generated using the default
%   methods defined in the functions default_D, default_H, and default_x_d.
%
%   p = SP_model(mpc, est_dyn) will use the functions specified in struct
%   est_dyn (function handles est_dyn.x_d, est_dyn.H, est_dyn.D) to
%   estimate missing dynamic parameters.
%
%   [p, details] = SP_model(...) will also return struct details, which has
%   the following fields:
%
%      Y_SP: The admittance matrix for the SP model
%        Y0: Admittance matrix of the physical network
%         E: Column vector of (complex) internal voltage of generators
%       x_d: Column vector of generator transient reactances
%       mpc: Struct mpc from the input after the power flow calculation
%
%   The admittance matrix details.Y0 will be computed by MATPOWER. 
%
%   References:
%
%   [1] T. Nishikawa and A. E. Motter, Comparative analysis of existing
%   models for power-grid synchronization, New J. Phys. 17, 015012 (2015).
%
%   [2] A. E. Motter, S. A. Myers, M. Anghel, and T. Nishikawa, Spontaneous
%   synchrony in power-grid networks, Nat. Phys. 9, 191-197 (2013).
%
%   See also default_D, default_H, default_x_d

%
% Copyright (C) 2015  Takashi Nishikawa
% 
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 2 of the License, or (at
% your option) any later version.
% 
% This program is distributed in the hope that it will be useful, but
% WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
% General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307,
% USA.

%   Last modified by Takashi Nishikawa on 1/27/2015

% The code below uses MATPOWER constants defined here.
define_constants

% Compute power flow using MATPOWER, suppressing output display.
mopt = mpoption( ...
    'OUT_ALL', 0, ...
    'VERBOSE', 0, ...
    'PF_DC', 0, ...
    'OUT_SYS_SUM', 0, ...
    'OUT_BUS', 0, 'OUT_BRANCH', 0);
mpc = runpf(mpc, mopt);
if ~mpc.success
    error('MATPOWER power flow calculation was not successful.')
end

% Number of generators
ngi = size(mpc.gen,1);

% Number of buses
n = size(mpc.bus,1);

% If the functions for estimating missing dynamic parameters are not given,
% use the default functions.
if nargin < 2
    est_dyn.x_d = @default_x_d;
    est_dyn.H = @default_H;
    est_dyn.D = @default_D;
end

% Use mpc.gen_dyn and mpc.load_dyn if they are given. Otherwise, generate
% empty place holder here to be filled in later with the default values.
% The columns are: x'_{d,i}, H_i, and D_i. Note that indexing for given
% gen_dyn should be 'external' one (and will be converted to 'internal'
% below, but converted back to 'external' before this function returns).
if ~isfield(mpc,'gen_dyn')    
%     disp('mpc.gen_dyn is missing and thus estimated.')
    mpc.gen_dyn = nan(ngi,3);
end
if ~isfield(mpc,'load_dyn')
    mpc.load_dyn = nan(n,1);
%     disp('mpc.load_dyn is missing and thus estimated.')
end

% Convert to MATPOWER's internal indexing scheme.  This is needed
% because makeYbus requires matpower internal indexing.  According to
% MATPOWER help file for ext2int, "all isolated buses, off-line generators
% and branches are removed along with any generators, branches or areas
% connected to isolated buses." "All buses are numbered consecutively,
% beginning at 1, and generators are reordered by bus number."
mpc = ext2int(mpc);
mpc = e2i_field(mpc, 'gen_dyn', 'gen');
mpc = e2i_field(mpc, 'load_dyn', 'bus');    

% Common system base
baseMVA = mpc.baseMVA;

% Angular system reference frequency omega_R (in radian)
if isfield(mpc, 'ref_freq')
    omega_R = 2*pi*mpc.ref_freq;
else
    p.omega_R = 2*pi*60;
    disp('mpc.ref_freq not defined. Assuming system ref frequency of 60 Hz.')
end
p.omega_R = omega_R;

% My indices for generator internal buses (gii) and generator terminal
% buses (gti), as well as the bus numbers (in mpc) for the terminal buses.
ngi = size(mpc.gen,1);
gtb = unique(mpc.gen(:,GEN_BUS));
ngt = length(gtb);

% Other (load) buses
is_gen = false(n,1);
is_gen(gtb) = true;
ltb = mpc.bus(~is_gen, BUS_I);
nl = length(ltb);

% Total number of nodes, including the generators' separate internal and
% terminal nodes, as well as load nodes.
N = ngi + ngt + nl;

%% Estimating missing dynamic parameters

% For generators:

% If x'_{d,i} is not given, use est_dyn.x_d to estimate it.
i = isnan(mpc.gen_dyn(:,1));
mpc.gen_dyn(i,1) = est_dyn.x_d(abs(mpc.gen(i,PG)));
x_d = mpc.gen_dyn(:,1);

% If H_i is not given, use est_dyn.H to estimate it.
i = isnan(mpc.gen_dyn(:,2));
mpc.gen_dyn(i,2) = est_dyn.H(abs(mpc.gen(i,PG)));
p.H = mpc.gen_dyn(:,2);

% If D_i is not given, use est_dyn.D to estimate it.
i = isnan(mpc.gen_dyn(:,3));
mpc.gen_dyn(i,3) = est_dyn.D(abs(mpc.gen(i,PG)));

% For loads:

% If D_i is not given, use est_dyn.D to estimate it.
i = isnan(mpc.load_dyn);
mpc.load_dyn(i) = est_dyn.D(abs(mpc.bus(i,PD)));

% The vector of constants D_i
p.D = [mpc.gen_dyn(:,3); mpc.load_dyn];

%% Computing E
% P and Q injected at generator terminals in p.u. on system base MVA.
Pi = mpc.gen(:,PG)/baseMVA;
Qi = mpc.gen(:,QG)/baseMVA;

% Voltage magnitude V and phase angle phi for the generator terminal buses
tb = mpc.gen(:,GEN_BUS); % (counting multiple generators)
V = mpc.bus(tb,VM);
phi = mpc.bus(tb,VA)/180*pi;
p.phi = phi;

% Compute the complex voltage E at the internal nodes of generators and
% motors.
E = ((V + Qi.*x_d./V) ...
    + 1j*(Pi.*x_d./V)).*exp(1j*phi);

%% Computing the admittance matrix Y_SP
Y0 = makeYbus(mpc);
Y0gg = Y0(gtb,gtb);
Y0ll = Y0(ltb,ltb);
Y0gl = Y0(gtb,ltb);
Y0lg = Y0(ltb,gtb);
Yd = sparse(1:ngi, 1:ngi, 1./(1j*x_d));
Y_SP = [
    Yd,             -Yd,        sparse(ngi,nl);
    -Yd,            Y0gg + Yd,  Y0gl;
    sparse(nl,ngi),  Y0lg,       Y0ll];

%% Computing the constants term A_i and the coupling matrix K_{ij}
Plg = mpc.bus(gtb,PD)/baseMVA;
Vg = mpc.bus(gtb,VM);
Pll = mpc.bus(ltb,PD)/baseMVA;
Vl = mpc.bus(ltb,VM);
p.A = [
    Pi(1:ngi);
    - Plg - Vg.^2.*real(diag(Y0gg + Yd));
    - Pll - Vl.^2.*real(diag(Y0ll))];
DE = sparse(1:N, 1:N, [abs(E); Vg; Vl]);
p.K = DE * abs(Y_SP) * DE;
p.gamma = angle(Y_SP) - pi/2.*(Y_SP ~= 0);

%% Fishing up
% Convert indexing for the power flow solution back to the original
% ordering.
mpc = i2e_field(mpc, 'gen_dyn', 'gen');
mpc = int2ext(mpc);

% Return intermediate parameters if asked.
if nargout > 1
    details.Y_SP = Y_SP;
    details.Y0 = Y0;
    details.E = E;
    details.x_d = x_d;
    details.mpc = mpc;
    details.ngi = ngi;
    details.ngt = ngt;
    details.nl = nl;
    details.V = [V; Vg; Vl];
    details.Pi = [Pi; -Plg; -Pll];
    % details.Qi = [Qi; Qlg; Qll];
end
