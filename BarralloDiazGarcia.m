%%  EXERCISE 1
% I. FORMULATION:
% (1) Decision variables:
%    e_ij = 1 if we move from city i to city j, 0 otherwise
% We assume that e=(e12, e13, e24, e34, e35, e45, e56, e64, e67, e47, e57)

% (2) Constraints:
%  - Signconstraints: 
%               e_ij =0,1  it is a binary variable
%  - Equality
%     e35=0 because we simly can´t go though five if previously passed 3
%     e12+e13=0
%     e47+e57+e67=0
%     e24+e34-e45-e47=0
%     e45-e57-e56=0
%     e56-e64-e67=0
%
%  - Inequalities
%     -e12+-e56<=-1
%     e34+e45<=1
%     e12+e34<=1
%     e13+e24<=1
%
% (3) Objective function:
%   z[min]=9e12+14e13+24e24+18e34+30e35+2e45+11e56+3e64+19e47+16e57+6e67
%
% II. CALCULATIONS
clear all;
c = [9 14 24 18 30 2 11 3 6 19 16];
A = [-1 0 0 0 0 0 -1 0 0 0 0
      0 0 0 1 0 1 0 0 0 0 0
      1 0 0 1 0 0 0 0 0 0 0
      0 1 1 0 0 0 0 0 0 0 0];
b = [-1 1 1 1];
lb = [0 0 0 0 0 0 0 0 0 0 0]; % zeros(11,1)
ub = [1 1 1 1 1 1 1 1 1 1 1]; % ones(11,1)
int = [1:11];
Aeq=[0 0 0 0 1 0 0 0 0 0 0
     1 1 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 1 1 1
     0 0 1 1 0 -1 0 0 0 -1 0
     0 0 0 0 0 1 -1 0 0 0 -1
     0 0 0 0 0 0 1 -1 -1 0 0];
beq=[0 1 1 0 0 0];
[vectx, benefit]=intlinprog(c,int,A,b,Aeq,beq,lb,ub);
 disp(benefit)
 disp(vectx)
%
fprintf("The best solution is:\n");

% III. SOLUTION 
% The results are:
% Minimum weight: 51
% (e12, e13, e24, e34, e35, e45, e56, e64, e67, e47, e57)
% ( 1 ,  0 ,  1 ,  0 ,  0 ,  1 ,  0 ,  0 ,  0 , 0  ,  1 )