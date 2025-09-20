---
layout: archive
title: ""   
permalink: /1BCS-CN/
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


## 1比特压缩感知
---
<div style="text-align:justify;">
1比特压缩感知（One-bit compressive sensing，1BCS）旨在从以下系统中恢复出稀疏信号 $\mathbf{x}\in\mathbb{R}^{n}$，
</div>

\begin{equation}
\mathbf{b} = \mathrm{Diag}(\mathbf{h}) \mathrm{sign}(\mathbf{A}\mathbf{x} + \boldsymbol{\varepsilon}) \tag{1bCS}
\end{equation} 

<div style="text-align:justify;">
其中，采样矩阵 $\mathbf{A}\in\mathbb{R}^{m\times n}$，观测向量 $\mathbf{b}$ 和符号翻转向量 $\mathbf{h}\in\{-1,1\}^{m}$，噪声 $\boldsymbol{\varepsilon}\in\mathbb{R}^{n}$，对角矩阵 $\mathrm{Diag}(\mathbf{h})$ 的对角元由  $\mathbf{h}$ 组成，符号函数定义为 $\mathrm{sign}(t)=1$ 当 $t>0$，否则 $\mathrm{sign}(t)=-1$。向量情形下，$\mathrm{sign}(\mathbf{x})=$ $(\mathrm{sign}(x_1)$ $\cdots$ $\mathrm{sign}(x_n))^\top$。值得一提的是，观测 $\mathbf{b}$ 是符号翻转后得到的，因而，问题更具挑战性。在该模型中，假设最多有 $k$ 个符号被翻转，即 $\mathbf{h}$ 满足 $\parallel\mathbf{h}-1\parallel_0\leq k$，其中 $k$ 为给定整数，零范数 $\parallel\mathbf{x}\parallel_0$ 表示 $\mathbf{x}$ 中非零元个数。为恢复信号，可求解以下优化模型。
</div> 
 <p style="line-height: 2;"></p>
 <div style="text-align:justify;"> 
◻️ 双稀疏约束优化模型     
\begin{equation}
\min_{\mathbf{x}\in\mathbb{R}^{n},\mathbf{z}\in\mathbb{R}^{m}}~  \parallel  \mathrm{Diag}(\mathbf{b}) \mathbf{A} \mathbf{x}+\mathbf{z} -\epsilon \parallel^2 + \eta \parallel \mathbf{x} \parallel^2,~~~\textrm{s.t.}~ \parallel\mathbf{x} \parallel_0\leq s,~ \parallel \mathbf{z}_+\parallel_0\leq k \tag{DSCO}
\end{equation}
<div style="text-align:justify;"> 
</div> 
◻️ 阶梯函数正则优化模型
\begin{equation}
\min_{\mathbf{x}\in\mathbb{R}^{n}}~  \sum_{i=1}^n (x_i^2+\varepsilon)^{q/2}+\lambda \parallel (\epsilon- \mathrm{Diag}(\mathbf{b}) \mathbf{A} \mathbf{x})_+ \parallel_0 \tag{SFRO}
\end{equation}
</div> 
<div style="text-align:justify;">
其中， 正整数 $s\ll n$ 和 $k\ll m$， 参数 $(\epsilon，\eta，\varepsilon，\lambda)>0$ 且 $\mathbf{z}_+=$ $(\max\{0,z_1\}$ $\cdots$ $\max\{0,z_m\})^\top$。度量 $\parallel \mathbf{z}_+\parallel_0$ 计算 $\mathbf{z}$ 中正元素的个数，它与阶梯（又称 0/1 损失）函数相关，其定义为 $\mathrm{step}(t)=1$ 当 $t>0$，否则 $\mathrm{step}(t)=0$。因此，$\|\mathbf{z}_+\|_0=$ $\mathrm{step}(z_1)+$ $\cdots$ $+\mathrm{step}(z_m)$。
</div> 

---

<div style="text-align:justify;">
程序包 - <a style="font-size: 16px; font-weight: bold;color:#006DB0" href="\files\OBCSpack-Matlab.zip" target="_blank">OBCSpack-Matlab</a> 和 <a style="font-size: 16px; font-weight: bold;color:#006DB0" href="\files\OBCSpack-Python.zip" target="_blank">OBCSpack-Python</a>（点击下载）提供了 2 个求解器。其核心算法分别来自以下 2 篇文章，其中， $\texttt{GPSP}$ 和 $\texttt{NM01}$ 分别用来求解模型 (DSCO) 和 (SFRO)。 
</div>  

> <b style="font-size:14px;color:#777777">GPSP</b> -<span style="font-size: 13.5px"> S Zhou, Z Luo, N Xiu, and G Li, Computing one-bit compressive sensing via double-sparsity constrained optimization, IEEE TSP, 70:1593-1608, 2022. </span>
<br> <b style="font-size:14px;color:#777777">NM01</b> -<span style="font-size: 14px"> S Zhou, L Pan, N Xiu, and H Qi, Quadratic convergence of smoothing Newton's method for 0/1 loss optimization, SIOPT, 31:3184–3211, 2021. </span>

---
<div style="text-align:justify;">
下面 Matlab 代码（Python 代码类似）展示了如何使用 $\texttt{OBCSpack}$ 求解 1BCS 问题。用户需输入数据 ($\texttt{A}$，$\texttt{b}$，$\texttt{s}$，$\texttt{k}$)，然后从 {'$\texttt{GPSP}$'，'$\texttt{NM01}$'} 中选择一个求解器进行求解。
</div>

<p style="line-height: 1;"></p>

```ruby
clc; close all; clear; addpath(genpath(pwd));

n      = 2000;                             % Signal dimension 
m      = ceil(0.5*n);                      % Number of measurements
s      = ceil(0.01*n);                     % Sparsity level
nf     = 0.05;                             % Noisy ratio
r      = 0.02;                             % Flipping ratio
k      = ceil(r*m);

A      = randn(m,n);
T      = randperm(n,s);
xo     = zeros(n,1);                      
xo(T)  = (1+rand(s,1)).*sign(randn(s,1));  
xo(T)  = xo(T)/norm(xo(T));                % True sparse solution
h      = ones(m,1);                        % Flipping vector
T      = randperm(m,k); 
h(T)   = -h(T);
b      = h.*sign(A(:,T)*xo(T)+nf*randn(m,1));; 

solver = {'GPSP','NM01'};
out    = OBCSpack(A,b,s,k,solver{1});  
fprintf(' Time:                  %6.3f sec\n',out.time);
fprintf(' Absolue error:         %6.2f %%\n', norm(xo-out.sol)*100);
fprintf(' Signal-to-noise ratio: %6.2f\n',-10*log10(norm(xo-out.sol)^2));
fprintf(' Hamming distence:      %6.3f\n',nnz(sign(A*out.sol)-b)/m)
```

<div style="text-align:justify;">
Matlab 版程序包 $\texttt{OBCSpack}$ 的输入和输出（Python 版的输入和输出类似）如下所示，其中输入 ($\texttt{A}$，$\texttt{b}$，$\texttt{s}$，$\texttt{k}$，$\texttt{solver}$) 为必需项。因为 $\texttt{NM01}$ 求解模型 (SFRO)，所以无需参数 $\texttt{s}$ 和 $\texttt{k}$。因此，如果选择 $\texttt{solver}$='$\texttt{NM01}$'，则当 $\texttt{s}$ 和 $\texttt{k}$ 未知时，可以将它们设置为 $\texttt{[]}$。参数 $\texttt{pars}$ 是可选的，但设置其中的一些参数可以提升求解器的性能和解的质量。
</div>

<p style="line-height: 1;"></p>

```ruby
function out = OBCSpack(A,b,s,k,solver,pars)
% -------------------------------------------------------------------------
% One-bit compressed sensing problem aims to recover a sparse signal x from
%
%                b = Diag(h).*sign( A*x + noise )
%
% 1) The double sparsity constrained optimization (DSCO)
%
%    min  ||Diag(b)*A*x+y-epsilon||^2 + eta||x||^2
%    s.t. ||x||_0<=s, ||y_+||_0<=k
%
% where (epsilon, eta)>0, s\in[1,n], k\in[0,m] are given.
%
% 2) The step function regularized optimization (SFRO)
%
%    min ||x.*x+vareps||^{q/2}_{q/2} + lam*||(epsilon - Diag(b)*A*x)_+||_0
%
% where (vareps, lam, epsilon)> 0, q\in(0,1).  
% -------------------------------------------------------------------------
% Inputs:
%  A:       The sensing matrix \in R^{m-by-n},                   (REQUIRED)
%  b:       The binary observation \in R^m, b_i\in{-1,1}         (REQUIRED)
%  s:       Sparsity level of x, an integer \in[1,n]             (REQUIRED)      
%  k:       An integer in [0,m], e.g., k = ceil(0.01m)           (REQUIRED)       
%  solver:  A text string, can be one of {'GPSP','NM01'}         (REQUIRED)            
%  pars:    Parameters are optional                              (OPTIONAL) 
%           ------------- For GPSP solving (DSCO)--------------------------
%           pars.eps     - The parameter in the model        (default 1e-4)
%           pars.eta     - The penalty parameter       (default 0.01/ln(n))
%           pars.acc     - Acceleration is used if acc=1        (default 0)
%           pars.big     - Start with a bigger s if big=1       (default 1)
%           pars.maxit   - Maximum number of iterations       (default 1e3) 
%           pars.tol     - Tolerance of halting condition    (default 1e-8)
%           -------------  For NM01 solving (SFRO)-------------------------
%           pars.x0      - The initial point           (default zeros(n,1))
%           pars.q       - Parameter in the objective         (default 0.5)
%           pars.vareps  - Parameter in the objective         (default 0.5)
%           pars.epsilon - Parameter in the objective        (default 0.15)
%           pars.lam     - The penalty parameter                (default 1)
%           pars.tau     - A useful parameter                   (default 1) 
%           pars.maxit   - Maximum number of iterations       (default 1e3)  
% -------------------------------------------------------------------------
% Outputs:
%     out.sol:   The sparse solution x
%     out.time:  CPU time
%     out.iter:  Number of iterations
%     out.obj:   Objective function value at out.sol 
% ------------------------------------------------------------------------
```
