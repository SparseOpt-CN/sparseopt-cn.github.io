---
layout: archive
title: ""   
permalink: /SRO-CN/
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


## 稀疏正则优化
---
<p style="line-height: 2;"></p>

\begin{equation}
\min_{\mathbf{x}\in\mathbb{R}^{n}} ~~  f(\mathbf{x}) + \lambda \parallel\mathbf{x}\parallel_0 \tag{SRO}
\end{equation}

<div style="text-align:justify;"> 
其中，函数 $f:\mathbb{R}^{n}\rightarrow \mathbb{R}$ 二次连续可微, $\lambda>0$ 是罚参数, $\parallel\mathbf{x}\parallel_0$ 代表了 $\mathbf{x}$ 的零范数，也即，计算 $\mathbf{x}$ 中非零元素个数。
</div>
 
---
<div style="text-align:justify;">
程序包 - <a style="font-size: 16px; font-weight: bold; color:#006DB0" href="\files\SROpack.zip" target="_blank">NL0R</a>（点击可直接下载）提供了 1 个求解器，其核心算法来自以下文章：
</div>

> <b style="font-size:14px;color:#777777">NL0R</b> - <span style="font-size: 14px"> S Zhou, L Pan, and N Xiu, Newton method for l0 regularized optimization, Numer Algorithms, 88:1541–1570, 2021. </span>

---
<div style="text-align:justify;">  
求解器 $\texttt{NL0R}$ 是二阶算法, 所以需要目标函数、梯度以及二阶海瑟矩阵子块。下面给出一个示例，展示如何定义函数来解决一个简单的稀疏约束优化（SCO）问题。其中，输入 $\texttt{x}$ 是变量，字符串变量 $\texttt{key}$ 用于指定计算内容：当 $\texttt{key}$='$\texttt{fg}$' 时表示计算目标函数值和梯度；当 $\texttt{key}$='$\texttt{h}$' 时表示计算海瑟矩阵子块，此时，海瑟矩阵子块由两个索引指标集 $\texttt{T1}$ 和 $\texttt{T2}$ 定义；如果只有一个输出，那么输出的子块包含了海瑟矩阵的 $\texttt{T1}$ 行和 $\texttt{T1}$ 列； 如果有两个输出，那么第一个输出子块包含了海瑟矩阵的 $\texttt{T1}$ 行和 $\texttt{T1}$ 列，第二个输出子块包含了海瑟矩阵的 $\texttt{T1}$ 行和 $\texttt{T2}$ 列。
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
对于以上简单的 SCO 问题，定义好函数后，就可以调用求解器 $\texttt{NL0R}$ 来求解该问题. 用户需要指定 ($\texttt{func}$, $\texttt{n}$, $\texttt{s}$)，必要时在 $\texttt{pars}$ 中设置一些参数，然后运行求解器。下面的代码展示了如何使用求解器 $\texttt{NL0R}$ 来求解该简单的 SCO 问题。
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
对于其他问题，用户可以通过修改 $\texttt{out1}$ 和 $\texttt{out2}$，但保持 $\texttt{switch}$ 的整体结构不变，来以类似方式定义相应的函数。作为示例，下面的代码给出了稀疏线性回归问题的函数定义。
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
在定义好稀疏线性回归问题的函数后，我们可以如下调用求解器 $\texttt{NL0R}$ 来求解该问题。
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
求解器 $\texttt{NL0R}$ 的输入与输出说明如下，其中输入参数 ($\texttt{func}$, $\texttt{n}$, $\texttt{lambda}$) 为必需项。$\texttt{pars}$ 中的参数为可选项，但设置某些参数可能会提升求解器的性能和解的质量。例如，调节合适的 $\texttt{pars.eta}$ 能显著改善求解器在收敛速度和精度方面的表现。
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
