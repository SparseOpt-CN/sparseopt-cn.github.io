---
layout: archive
title: ""   
permalink: /SRO/
author_profile: true
---

<style>
a:link {
  text-decoration: none;
}

a:visited {
  text-decoration: none;
}

a:hover {
  text-decoration: underline;
}

a:active {
  text-decoration: underline;
}
</style>


## Sparsity-regularized optimization
---
<p style="line-height: 2;"></p>

\begin{equation}
\min_{\mathbf{x}\in\mathbb{R}^{n}} ~~  f(\mathbf{x}) + \lambda \parallel\mathbf{x}\parallel_0 \tag{SRO}
\end{equation}

<div style="text-align:justify;"> 
where $f:\mathbb{R}^{n}\rightarrow \mathbb{R}$ is a continuously or twice continuously differentiable function, $\lambda>0$ is the penalty parameter, and   $\parallel\mathbf{x}\parallel_0$ denotes the L0 norm, which counts the number of nonzero entries in $\mathbf{x}$.
</div>
 
---
<div style="text-align:justify;">
The package can be downloaded here - <a style="font-size: 16px; font-weight: bold; color:#006DB0" href="\files\SROpack.zip" target="_blank">NL0R</a>, which is developed based on the algorithm proposed in the following paper:
</div>

> <b style="font-size:14px;color:#777777">NL0R</b> - <span style="font-size: 14px"> S Zhou, L Pan, and N Xiu, Newton method for l0 regularized optimization, Numer Algorithms, 88:1541–1570, 2021. </span>

---
<div style="text-align:justify;">  
Note that $\texttt{NL0R}$ is a second-order method, requiring the objective, gradient, and sub-Hessian of $f$. Below is an example of how to define these functions for the solver to solve a simple problem, where input $\texttt{x}$ is the variable, string variable $\texttt{key}$ specifies the computation: $\texttt{key}$='$\texttt{fg}$' for the objective value and the gradient, and $\texttt{key}$='$\texttt{h}$' for the sub-Hessian decided by two indix sets $\texttt{T1}$ and $\texttt{T2}$.  Specifically, when $\texttt{key}$ = '$\texttt{fg}$', if there is only one output, it returns the objective function value; if there are two outputs, the first is the objective function value and the second is the gradient. When $\texttt{key}$ = '$\texttt{h}$', if there is only one output, it returns the sub-Hessian containing the $\texttt{T1}$ rows and $\texttt{T1}$ columns of the Hessian; if there are two outputs, the first corresponds to the sub-Hessian containing the $\texttt{T1}$ rows and $\texttt{T1}$ columns of the Hessian, and the second corresponds to the sub-Hessian containing the $\texttt{T1}$ rows and $\texttt{T2}$ columns of the Hessian.
</div>
<p style="line-height: 1;"></p>

```ruby
function [out1,out2] = funcSimpleEx(x,key,T1,T2)
    % This code provides information for
    %     min   x'*[6 5;5 8]*x+[1 9]*x-sqrt(x'*x+1)  
    a   = sqrt(sum(x.*x)+1);
    switch key
        case 'fg'    
            out1 = x'*[6 5;5 8]*x+[1 9]*x-a;       % objective
            if  nargout == 2 
                out2 = 2*[6 5;5 8]*x+[1; 9]-x./a;  % gradient
            end
        case 'h'
            H   = 2*[6 5;5 8]+(x*x'-a*eye(2))/a^3; % sub-Hessian formed by rows indexed by T1 and columns indexed by T1
            out1 = H(T1,T1);
            if  nargout == 2 
                out2 = H(T1,T2);                   % sub-Hessian formed by rows indexed by T1 and columns indexed by T2
            end
    end
end
```

<div style="text-align:justify;">
After defining the functions for the simple problem, one can call $\texttt{NL0R}$ to solve it. Users need to specify ($\texttt{func}$, $\texttt{n}$, $\texttt{lambda}$), set some parameters in $\texttt{pars}$ if necessary, and then run the solver. The following codes demonstrate $\texttt{NL0R}$ to solve this simple problem.
</div>
<p style="line-height: 1;"></p>

```ruby
% demon a simple SRO problem
clc; close all; clear all;  addpath(genpath(pwd));

n        = 2;
lambda   = 0.5;
pars.eta = 0.1;
out      = NL0R(@funcSimpleEx,n,lambda,pars); 
fprintf(' Objective:      %.4f\n', out.obj); 
fprintf(' CPU time:      %.3fsec\n', out.time);
fprintf(' Iterations:        %4d\n', out.iter);
```

<div style="text-align:justify;">
For other problems, users can similarly define the functions by modifying $\texttt{out1}$ and $\texttt{out2}$ while preserving the overall structure of $\texttt{switch}$. As an illustration, the following codes define the functions of a sparse linear regression problem.
</div>
<p style="line-height: 1;"></p>

```ruby
function [out1,out2] = funcLinReg(x,key,T1,T2,A,b)
    % This code provides information for
    %     min   0.5*||Ax-b||^2 
    % where A in R^{m x n} and b in R^{m x 1}    
    switch key
        case 'fg'
            Tx   = find(x~=0);
            Axb  = A(:,Tx)*x(Tx)-b;
            out1 = (Axb'*Axb)/2;      % objective 
            if  nargout == 2 
                out2 = (Axb'*A)';     % gradient 
            end
        case 'h'        
            AT   = A(:,T1); 
            out1 = AT'*AT;            %sub-Hessian formed by rows indexed by T1 and columns indexed by T1   
            if  nargout == 2
                out2 = AT'*A(:,T2);   %sub-Hessian formed by rows indexed by T1 and columns indexed by T2
            end       
    end
end
```

<div style="text-align:justify;">
After defining the functions of the sparse linear regression problem, we call $\texttt{NL0R}$ to solve the problem as follows.
</div>
<p style="line-height: 1;"></p>

```ruby
% demon sparse linear regression problems 
clc; close all; clear all; addpath(genpath(pwd));

n        = 2000;  
m        = ceil(0.25*n); 
s        = ceil(0.05*n);

Tx       = randperm(n,s);  
xopt     = zeros(n,1);  
xopt(Tx) = (0.25+rand(s,1)).*sign(randn(s,1)); 
A        = randn(m,n)/sqrt(m); 
b        = A*xopt;  
func     = @(x,key,T1,T2)funcLinReg(x,key,T1,T2,A,b);

lambda   = 0.01;
pars.eta = 1.0;
out      = NL0R(func,n,lambda,pars); 
```

<div style="text-align:justify;">
The inputs and outputs of $\texttt{NL0R}$ are detailed below, where inputs ($\texttt{func}$, $\texttt{n}$, $\texttt{lambda}$) are required. The parameters in $\texttt{pars}$ are optional, but setting certain ones may improve the solver's performance and the solution quality. For example, tuning a proper $\texttt{pars.eta}$ may significantly improve the solver performance in terms of convergence speed and accuracy.
</div>

<p style="line-height: 1;"></p>

```ruby
function out = NL0R(func,n,lambda,pars)
% -------------------------------------------------------------------------
% This code aims at solving the L0 norm regularized optimization 
%
%         min_{x\in R^n} f(x) + lambda*||x||_0^0
%
% where f: R^n->R, lambda>0
% ||x||_0^0 counts the number of non-zero entries
% -------------------------------------------------------------------------
% Inputs:
%   func:   A function handle defines                            (REQUIRED)
%                    (objective, gradient, sub-Hessian)
%   n:      Dimension of the solution x                          (REQUIRED) 
%   lambda: The penalty parameter                                (REQUIRED)  
%   pars  : All parameters are OPTIONAL
%           pars.x0     -- Starting point of x         (default zeros(n,1))
%           pars.tol    -- Tolerance of halting conditions   (default 1e-6)
%           pars.maxit  -- Maximum number of iterations      (default  2e3) 
%           pars.uppf   -- An upper bound of final objective (default -Inf)
%                          Useful for noisy case 
%           pars.eta    -- A positive scalar                    (default 1)  
%                          Tuning it may improve solution quality
%           pars.update -- =1 update penalty parameter lambda   (default 1)
%                          =0 fix penalty parameter lambda
%           pars.disp   -- =1 show results for each step        (default 1)
%                          =0 not show results for each step
% -------------------------------------------------------------------------
% Outputs:
%   out.sol :  The sparse solution x
%   out.obj :  Objective function value at out.sol 
%   out.iter:  Number of iterations
%   out.time:  CPU time
% -------------------------------------------------------------------------
% Send your comments and suggestions to <<< slzhou2021@163.com >>>   
% WARNING: Accuracy may not be guaranteed!!!!!  
% -------------------------------------------------------------------------
```
