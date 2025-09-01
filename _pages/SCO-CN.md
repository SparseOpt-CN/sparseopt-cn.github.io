---
layout: archive
title: ""   
permalink: /SCO-CN/
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

 
##  稀疏约束优化 
---

<p style="line-height: 1;"></p>
\begin{equation}
\min_{\mathbf{x}\in\mathbb{R}^{n}} ~~  f(\mathbf{x}),~~~~ \mbox{s.t.}~~ \parallel\mathbf{x}\parallel_0\leq s  \tag{SCO}
\end{equation}

<div style="text-align:justify;">
其中，函数 $f:\mathbb{R}^{n}\rightarrow \mathbb{R}$ 连续可微，最好二次连续可微，正整数 $s\ll n$，零范数 $\|\mathbf{x}\|_0$ 计算 $\mathbf{x}$ 中非零元个数。
</div>
 
<!-- ## <span style="color:#8C8C8C"> The solver and its demonstration </span> -->

---
<div style="text-align:justify;"> 
程序包 - <a style="font-size: 16px; font-weight: bold;color:#006DB0" href="\files\SCOpack-Matlab.zip" target="_blank">SCOpack-Matlab</a>（点击下载）提供了 3 个求解器，其核心算法分别来自以下 3 篇文章：

</div>

> <b style="font-size:14px;color:#777777">NHTP</b> - <span style="font-size: 14px"> S Zhou, N Xiu, and H Qi, Global and quadratic convergence of Newton hard-thresholding pursuit, JMLR, 22:1-45, 2021. </span>
<br><b style="font-size:14px;color:#777777">GPNP</b> - <span style="font-size: 14px"> S Zhou, Gradient projection Newton pursuit for sparsity constrained optimization, ACHA, 61:75-100, 2022. </span>
<br><b style="font-size:14px;color:#777777">IIHT</b> - <span style="font-size: 14px"> L Pan, S Zhou, N Xiu, and H Qi, A convergent iterative hard thresholding for nonnegative sparsity optimization, PJO, 13:325-353, 2017. </span>

---
<div style="text-align:justify;">  
求解器 $\texttt{NHTP}$ 和 $\texttt{GPNP}$ 是基于二阶算法开发的, 所以需要目标函数、梯度以及海瑟矩阵子块，而 $\texttt{IIHT}$ 属于一阶方法，仅需要目标函数和梯度。基于 Matlab 语言（基于 Python 语言，可进行类似的定义）下面给出一个示例，展示如何以统一的方式为三个求解器定义函数来求解一个简单的稀疏约束优化（SCO）问题。其中，句柄函数 $\texttt{funcSimpleEx}$ 的输入中，$\texttt{x}$ 是自变量，$\texttt{key}$ 是字符串变量，$\texttt{T1}$ 和 $\texttt{T2}$ 为两个索引指标集。这里，$\texttt{key}$ 用于指定计算内容：当 $\texttt{key}$='$\texttt{fg}$' 时表示计算目标函数值和梯度，此时，若只有一个输出，则输出目标函数值，若有两个输出，则第一个输出为目标函数值，第二个输出为梯度；当 $\texttt{key}$='$\texttt{h}$' 时表示计算海瑟矩阵子块，此时，海瑟矩阵子块由两个索引指标集 $\texttt{T1}$ 和 $\texttt{T2}$ 定义，若只有一个输出，则输出的子块包含海瑟矩阵的 $\texttt{T1}$ 行和 $\texttt{T1}$ 列，若有两个输出，则第一个输出子块包含海瑟矩阵的 $\texttt{T1}$ 行和 $\texttt{T1}$ 列，第二个输出子块包含海瑟矩阵的 $\texttt{T1}$ 行和 $\texttt{T2}$ 列。
</div>
<p style="line-height: 1;"></p>

```ruby
function [out1,out2] = funcSimpleEx(x,key,T1,T2)
    % This code provides information for
    %     min   x'*[6 5;5 8]*x+[1 9]*x-sqrt(x'*x+1) 
    %     s.t. \|x\|_0<=s
    % where n=2 and s=1   
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
对于以上一个简单的 SCO 问题，定义好函数后，就可以调用程序包 $\texttt{SCOpack-Matlab}$ 来求解该问题。用户需指定 ($\texttt{func}$, $\texttt{n}$, $\texttt{s}$)，再从 {'$\texttt{NHTP}$', '$\texttt{GPNP}$', '$\texttt{IIHT}$'} 中选择一个求解器名称，必要时在 $\texttt{pars}$ 中设置一些参数，然后运行求解器。下面的代码展示了如何使用 $\texttt{SCOpack-Matlab}$ 来求解该简单的 SCO 问题。
</div>
<p style="line-height: 1;"></p>

```ruby
% demon a simple sparsity constrained problem
clc; close all; clear all;  addpath(genpath(pwd));

n        = 2;
s        = 1; 
func     = @funcSimpleEx;
solver   = {'NHTP','GPNP','IIHT'};
pars.eta = 0.1; % useful for 'NHTP'
out      = SCOpack(func,n,s,solver{2},pars); 

fprintf(' Objective:      %.4f\n', out.obj); 
fprintf(' CPU time:      %.3fsec\n', out.time);
fprintf(' Iterations:        %4d\n', out.iter);
```

<div style="text-align:justify;">
对于其他问题，用户可以通过修改 $\texttt{out1}$ 和 $\texttt{out2}$，但保持 $\texttt{switch}$ 的整体结构不变，来以类似方式定义相应的函数。作为示例，下面的 Matlab 代码给出了稀疏线性回归问题的函数定义。
</div>
<p style="line-height: 1;"></p>

```ruby
function [out1,out2] = funcLinReg(x,key,T1,T2,A,b)
    % This code provides information for
    %     min   0.5*||Ax-b||^2 
    %     s.t. \|x\|_0<=s
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
在定义好稀疏线性回归问题的函数后，我们可以如下调用程序包 $\texttt{SCOpack-Matlab}$ 来求解该问题。
</div>
<p style="line-height: 1;"></p>

```ruby
% demon sparse linear regression problems 
clc; close all; clear all; addpath(genpath(pwd));

n        = 20000;  
m        = ceil(0.25*n); 
s        = ceil(0.025*n);

Tx       = randperm(n,s);  
xopt     = zeros(n,1);  
xopt(Tx) = randn(s,1); 
A        = randn(m,n)/sqrt(m); 
b        = A*xopt;  

func     = @(x,key,T1,T2)funcLinReg(x,key,T1,T2,A,b);
pars.tol = 1e-6;
solver   = {'NHTP','GPNP','IIHT'};
out      = SCOpack(func,n,s,solver{2},pars);
PlotRecovery(xopt,out.sol,[900,500,500,250],1)
```

<div style="text-align:justify;">
程序包 $\texttt{SCOpack-Matlab}$ 的输入与输出（Python 版本的输入和输出类似）说明如下，其中输入参数 ($\texttt{func}$, $\texttt{n}$, $\texttt{s}$, $\texttt{solvername}$) 为必需项。$\texttt{pars}$ 中的参数为可选项，但设置某些参数可能会提升求解器的性能和解的质量。例如，当选择求解器 $\texttt{NHTP}$ 时，调节合适的 $\texttt{pars.eta}$ 能显著改善求解器在收敛速度和精度方面的表现。此外，求解器 $\texttt{IIHT}$ 还支持带非负约束的 SCO 问题，只需设置 $\texttt{pars.neg}$=1 即可。
</div>

<p style="line-height: 1;"></p>

```ruby
function out = SCOpack(func,n,s,solvername,pars)
% -------------------------------------------------------------------------
% This code aims at solving the sparsity constrained optimization (SCO),
%
%         min_{x\in R^n} f(x),  s.t. ||x||_0<=s
%
% or non-negative and sparsity constrained optimization (NSCO):
%
%         min_{x\in R^n} f(x),  s.t. ||x||_0<=s, x>=0 
%
% where f: R^n->R and s<<n is an integer.
% -------------------------------------------------------------------------
% Inputs:
%   func:   A function handle defines                            (REQUIRED)
%                    (objective, gradient, sub-Hessian)
%   n:      Dimension of the solution x                          (REQUIRED)
%   s:      Sparsity level of x, an integer between 1 and n-1    (REQUIRED)
%   solver: A text string, can be one of {'NHTP','GPNP','IIHT'}  (REQUIRED)
%   pars  : ---------------For all solvers --------------------------------
%           pars.x0    --  Starting point of x         (default zeros(n,1))
%           pars.disp  --  =1 show results for each step        (default 1)
%                          =0 not show results for each step
%           pars.maxit --  Maximum number of iterations      (default  2e3) 
%           pars.tol   --  Tolerance of halting conditions   (default 1e-6)
%           pars.uppf  --  An upper bound of final objective (default -Inf)
%                          Useful for noisy case
%           ---------------Particular for NHTP ----------------------------
%           pars.eta   --  A positive scalar                    (default 1)  
%                          Tuning it may improve solution quality 
%           ---------------Particular for IIHT ----------------------------
%           pars.neg   --  =0 for model (SCO)                   (default 1)
%                          =1 for model (NSCO)
% -------------------------------------------------------------------------
% Outputs:
%     out.sol :   The sparse solution x
%     out.obj :   Objective function value at out.sol 
%     out.iter:   Number of iterations
%     out.time:   CPU time
% -------------------------------------------------------------------------
% Send your comments and suggestions to <<< slzhou2021@163.com >>>   
% WARNING: Accuracy may not be guaranteed!!!!!  
% -------------------------------------------------------------------------
```
