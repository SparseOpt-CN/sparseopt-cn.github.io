---
layout: archive
title: ""   
permalink: /CS-CN/
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


## 压缩感知
---
<div style="text-align:justify;">
压缩感知（Compressive sensing，CS）问题旨在从如下线性系统中恢复出稀疏信号 $\mathbf{x}\in\mathbb{R}^{n}$，
</div>

\begin{equation}
\mathbf{b} = \mathbf{A}\mathbf{x} + \boldsymbol{\varepsilon} \tag{CS}
\end{equation} 

<div style="text-align:justify;">
其中，采样矩阵 $\mathbf{A}\in\mathbb{R}^{m\times n}$，观测向量 $\mathbf{b}\in\mathbb{R}^{m}$， 噪音 $\boldsymbol{\varepsilon}\in\mathbb{R}^{n}$。为了恢复信号，常用的优化模型包括：
</div>   

<p style="line-height: 2;"></p>

<div style="text-align:justify;">
◻️ 稀疏约束 CS 模型
\begin{equation}
\min_{\mathbf{x}\in\mathbb{R}^{n}}~ \frac{1}{2}\parallel\mathbf{A}\mathbf{x}-\mathbf{b} \parallel^2,~~~\textrm{s.t.}~ \parallel\mathbf{x} \parallel_0\leq s \tag{SCCS}
\end{equation}
</div> 

<div style="text-align:justify;">
◻️ L0 和 Lq (0<q<1) 范数正则 CS 模型
\begin{equation}
\min_{\mathbf{x}\in\mathbb{R}^{n}}~ \frac{1}{2}\parallel\mathbf{A}\mathbf{x}-\mathbf{b} \parallel^2+\lambda \parallel\mathbf{x} \parallel_q^q \tag{Lq-RCS}
\end{equation}
</div>   

<div style="text-align:justify;">
◻️ 加权 L1 范数正则 CS 模型
\begin{equation}
\min_{\mathbf{x}\in\mathbb{R}^{n}}~ \frac{1}{2}\parallel\mathbf{A}\mathbf{x}-\mathbf{b} \parallel^2+\lambda \parallel \mathbf{W} \mathbf{x} \parallel_1 \tag{RL1CS}
\end{equation} 
其中，L0 (零）范数 $\parallel\mathbf{x}\parallel_0$ 计算 $\mathbf{x}$ 中非零元素的个数，Lq 范数 $\parallel\mathbf{x}\parallel_q^q=|x_1|^q+\cdots+|x_n|^q$，0<q<1，罚参数 $\lambda>0$，对角矩阵 $\mathbf{W}$ 的对角元都为正数。
</div> 
---
<div style="text-align:justify;">
程序包 - <a style="font-size: 16px; font-weight: bold;color:#006DB0" href="\files\CSpack-Matlab.zip" target="_blank">CSpack-Matlab</a>（点击下载）提供了 6 个求解器。其核心算法分别来自以下 6 篇文章，其中，$\texttt{NHTP}$、  $\texttt{GPSP}$ 和 $\texttt{IIHT}$ 适用于模型 (SCCS)，$\texttt{PSNP}$ 适用于模型 (L0-RCS) 和 (Lq-RCS) (0<q<1)，$\texttt{NL0R}$ 适用于模型 (L0-RCS)，$\texttt{MIRL1}$ 适用于模型 (RL1CS)。
</div>  
> <b style="font-size:14px;color:#777777">NHTP</b> - <span style="font-size: 14px"> S Zhou, N Xiu, and H Qi, Global and quadratic convergence of Newton hard-thresholding pursuit, JMLR, 22:1-45, 2021. </span>
<br><b style="font-size:14px;color:#777777">GPSP</b> - <span style="font-size: 14px"> S Zhou, Gradient projection Newton pursuit for sparsity constrained optimization, ACHA, 61:75-100, 2022. </span>
<br><b style="font-size:14px;color:#777777">PSNP</b> - <span style="font-size: 14px"> S Zhou, X Xiu, Y Wang, and D Peng, Revisiting Lq ( 0 ≤ q < 1 ) norm regularized optimization, arXiv:2306.14394, 2023. </span>
<br><b style="font-size:14px;color:#777777">NL0R</b> - <span style="font-size: 14px"> S Zhou, L Pan, and N Xiu, Newton method for l0 regularized optimization, Numer Algorithms, 88:1541–1570, 2021. </span>
<br><b style="font-size:14px;color:#777777">IIHT</b> - <span style="font-size: 14px"> L Pan, S Zhou, N Xiu, and H Qi, A convergent iterative hard thresholding for nonnegative sparsity optimization, PJO, 13:325-353, 2017. </span>
<br><b style="font-size:14px;color:#777777">MIRL1</b> - <span style="font-size: 14px"> S Zhou, N Xiu, et. al, A Null-space-based weighted l1 minimization approach to compressed sensing, Inf inference, 5:76-102, 2016. </span>

---
<div style="text-align:justify;">
基于 Matlab 语言（基于Python 语言，情况类似）以下代码展示了如何使用 $\texttt{CSpack}$ 来求解 CS 问题。用户需输入数据 ($\texttt{A}$，$\texttt{At}$，$\texttt{b}$，$\texttt{n}$，$\texttt{s}$) 并从 {'$\texttt{NHTP}$'，'$\texttt{GPNP}$'，'$\texttt{IIHT}$'，'$\texttt{PSNP}$'，'$\texttt{NL0R}$'，'$\texttt{MILR1}$'} 中选择一个作为求解器，然后运行求解。 
</div>

<p style="line-height: 1;"></p>

```ruby
clc; clear; close all; addpath(genpath(pwd));

n       = 10000;  
m       = ceil(0.25*n); 
s       = ceil(0.05*n); 

T       = randperm(n,s);  
xopt    = zeros(n,1);
xopt(T) = (0.1+rand(s,1)).*sign(randn(s,1));  
A       = randn(m,n)/sqrt(m);   
b       = A(:,T)*xopt(T)+0.00*randn(m,1);  

solver  = {'NHTP', 'GPNP', 'IIHT', 'PSNP', 'NL0R', 'MIRL1'};
out     = CSpack(A,[],b,n,s,solver{1}); 

fprintf(' Objective of xopt:       %.2e\n', norm(A*xopt-b)^2/2);
fprintf(' Objective of out.sol:    %.2e\n', out.obj);
fprintf(' Sparsity of out.sol:     %2d\n', nnz(out.sol));
fprintf(' Computational time:      %.3fsec\n',out.time); 
```

<div style="text-align:justify;">
程序包 $\texttt{CSpack-Matlab}$ 的输入和输出（Python 版本的输入和输出类似）如下所示，其中输入 ($\texttt{A}$，$\texttt{At}$，$\texttt{b}$，$\texttt{n}$，$\texttt{s}$，$\texttt{solver}$) 为必需项。如果 $\texttt{A}$ 是一个矩阵，则 $\texttt{At}$ 可以为其转置 $\texttt{A}'$ 也可以为空 $\texttt{[]}$。 如果 $\texttt{A}$ 是函数句柄，则必须提供 $\texttt{At}$。如果 $\texttt{solver}$ 为 {'$\texttt{PSNP}$'，'$\texttt{NL0R}$'，'$\texttt{MILR1}$'} 之一，则 $\texttt{s}$ 可设置为 $\texttt{[]}$。 如果 $\texttt{solver}$ 为 {'$\texttt{NHTP}$'，'$\texttt{GPNP}$'，'$\texttt{IIHT}$'} 之一，则必须提供 $\texttt{s}$。结构体 $\texttt{pars}$ 中的参数是可选的，但设置其中的一些参数可以提升求解器的性能和解的质量。
</div>

<p style="line-height: 1;"></p>

```ruby
function out = CSpack(A,At,b,n,s,solver,pars)
% -------------------------------------------------------------------------
% This solver solves compressive sensing (CS) in one of the following forms
%
% 1) The sparsity constrained compressive sensing (SCCS)
%
%         min_{x\in R^n} 0.5||Ax-b||^2  s.t. ||x||_0<=s
%
% 2) The L0 regularized compressive sensing (LqCS)
%
%         min_{x\in R^n} 0.5||Ax-b||^2 + lambda * ||x||_q^q,  0<=q<1 
%
% 3) The reweighted L1-regularized compressive sensing (RL1CS)
%
%         min_{x\in R^n} 0.5||Ax-b||^2 + lambda||Wx||_1
%
% where s << n is the given sparsity and lambda>0 
%       A\in\R{m by n} the measurement matrix
%       b\in\R{m by 1} the observation vector 
%       W\in\R{n by n} is a diagonal matrix to be updated iteratively
% -------------------------------------------------------------------------
% Inputs:
%   A  :     The measurement matrix, A\in\R{m by n}              (REQUIRED)
%   At :     The transpose of A and can be [] if A is a matrix   (REQUIRED)
%            But At is REQUIRED if A is a function handle 
%            i.e., A*x = A(x); A'*y = At(y); 
%   b:       The observation vector  b\in\R{m by 1}              (REQUIRED)
%   n:       Dimension of the solution x,                        (REQUIRED)
%   s:       The sparsity level, if unknown, put it as []        (REQUIRED)
%   solver:  A text string, can be one of                        (REQUIRED)
%            {'NHTP','GPNP','PSNP','NL0R','IIHT','MILR1'}
%
%           --------------------------------------------------------------------------------
%                    |  'NHTP'   |  'GPNP'   |  'PSNP'   |  'NL0R'   |  'IIHT'   |  'MIRL1'   
%           --------------------------------------------------------------------------------
%           Model    |   SCCS    |   SCCS    |   LqRCS   |   L0RCS   |   SCCS    |   RL1CS     
%           Method   | 2nd-order | 2nd-order | 2nd-order | 2nd-order | 1st-order | 1st-order  
%           Sparsity | required  | required  |  no need  |  no need  | required  |  no need
%           --------------------------------------------------------------------------------  
%
%   pars:   All parameters are optional                          (OPTIONAL)
%           ----------------For all solvers -------------------------------
%           pars.x0     --  Starting point of x       (default, zeros(n,1))                     
%           pars.disp   --  =1, show results for each step      (default,1)
%                           =0, not show results for each step
%           pars.maxit  --  Maximum number of iterations     (default, 2e3) 
%           pars.tol    --  Tolerance of stopping criteria   (default,1e-6)
%           ----------------Particular for NHTP ---------------------------
%           pars.eta    --  A positive scalar for 'NHTP'       (default, 1)  
%                           Tuning pars.eta may improve solution quality.
%           ----------------Particular for PSNP ---------------------------
%           pars.q      --  Decide Lq norm                  (default,  0.5)  
%           pars.lambda --  An initial penalty parameter    (default,  0.1)
%           pars.obj    --  A predefined lower bound of f(x)(default,1e-20)
%           ----------------Particular for NL0R ---------------------------
%           pars.tau    --  A positive scalar for 'NL0R'    (default,    1)  
%           pars.lambda --  An initial penalty parameter    (default,  0.1)
%           pars.obj    --  A predefined lower bound of f(x)(default,1e-20)
%           ----------------Particular for IIHT ---------------------------
%           pars.neg    --  =0, Compute SCCS without x>=0       (default,0)
%                           =1, Compute SCCS with x>=0
% -------------------------------------------------------------------------
% Outputs:
%     out.sol:   The sparse solution x
%     out.sp:    Sparsity level of out.sol
%     out.time:  CPU time
%     out.iter:  Number of iterations
%     out.obj:   Objective function value at out.sol 
% -------------------------------------------------------------------------
```
