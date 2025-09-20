---
layout: archive
title: ""   
permalink: /SQCQP-CN/
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


## 稀疏二次约束二次规划
---
\begin{equation}
\begin{array}{rl}
\min\limits_{\mathbf{x}\in\mathbb{R}^{n}} &  0.5\mathbf{x}^{\top}\mathbf{Q}_0\mathbf{x}+\mathbf{q}_0^{\top}\mathbf{x}\\\\\\
\mbox{s.t.} & 0.5\mathbf{x}^{\top}\mathbf{Q}_i\mathbf{x}+\mathbf{q}^{\top}_i\mathbf{x}+c_i\leq0,~i=1,2,\ldots,k\\\\\\
&\mathbf{A}\mathbf{x}\leq \mathbf{b},~~\mathbf{B}\mathbf{x} = \mathbf{d}\\\\\\
& l\leq x_i \leq u,~i=1,2,\ldots,n\\\\\\
&\parallel\mathbf{x}\parallel_0\leq s
\end{array} \tag{SQCQP}
\end{equation}

其中参数定义如下：
- 矩阵 $\mathbf{Q}_i\in\mathbb{R}^{n\times n}$，向量 $\mathbf{q}_i\in\mathbb{R}^{n}$，数 $c_i\in\mathbb{R}$，$i=0,1,\ldots,k$
- 矩阵 $\mathbf{A}\in\mathbb{R}^{m_1\times n}$，向量 $\mathbf{b}\in\mathbb{R}^{m_1}$
- 矩阵 $\mathbf{B}\in\mathbb{R}^{m_2\times n}$，向量 $\mathbf{d}\in\mathbb{R}^{m_2}$
- 下界 $l$ 和 上界 $u$ 满足 $l \leq 0 \leq u$
- 零范数 $\parallel\mathbf{x}\parallel_0$ 计算 $\mathbf{x}$ 中非零元个数
- 正整数 $s\ll n$
         
<!--## <span style="color:#8C8C8C"> The solver and its demonstration </span> -->
--- 
<div style="text-align:justify;">
程序包 - <a style="font-size: 16px; font-weight: bold;color:#006DB0" href="\files\SQCQPpack-Matlab.zip" target="_blank">SQSQPpack-Matlab</a> 和 <a style="font-size: 16px; font-weight: bold;color:#006DB0" href="\files\SQCQPpack-Python.zip" target="_blank">SQCQPpack-Python</a>（点击下载）提供了 1 个求解器，其核心算法来自以下文章：
</div>

> <b style="font-size:14px;color:#777777">SNSQP</b> - <span style="font-size: 14px"> S Li, S  Zhou, Z  Luo, Sparse quadratically constrained quadratic programming via semismooth Newton method, arXiv:2503.15109, 2025. </span>

---
<div style="text-align:justify;">
以下 Matlab 代码（Python 代码类似）展示了该求解器在稀疏投资组合选择问题中的一个应用示例。用户需要指定输入 ($\texttt{n}$, $\texttt{s}$, $\texttt{Q0}$, $\texttt{q0}$, $\texttt{Qi}$, $\texttt{qi}$, $\texttt{ci}$, $\texttt{ineqA}$, $\texttt{ineqb}$, $\texttt{eqA}$, $\texttt{eqb}$, $\texttt{lb}$, $\texttt{ub}$)。如果某个输入不需要，则将其设置为 $\texttt{[ ]}$。
</div>

<p style="line-height: 1;"></p>

```ruby
% demon sparse portfolio selection problems
clc; clear all; close all;  addpath(genpath(pwd));

n     = 1000;
s     = 10;

B     = 0.01 * rand(ceil(n/4),n);
D     = diag(0.01*rand(n,1));
Q0    = 2*( B'*B + D);
q0    = zeros(n,1); 
Qi    = cell(1,1);
Qi{1} = 2*D;
qi    = zeros(n,1);
ci    = -0.001;
ineqA = -0.5*randn(1,n);
ineqb = -0.002;
eqA   = ones(1,n);
eqb   = 1;
lb    = 0;
ub    = 0.3;
    
pars.x0       = ((lb+ub)/2).*ones(n,1);
pars.tau      = 1; % decrease this value if the algorithm does not converge
pars.dualquad = 0*ones(length(ci));
pars.dualineq = 0.001*ones(length(ineqb)); 
pars.dualeq   = 0.001*ones(length(eqb));
Out           = SNSQP(n,s,Q0,q0,Qi,qi,ci,ineqA,ineqb,eqA,eqb,lb,ub,pars);
```

<div style="text-align:justify;">
Matlab 版求解器 $\texttt{SNSQP}$ 的输入和输出（Python 版的输入和输出类似）如下所示。注意，输入 $\texttt{Qi}$ 是一个单元，包含了 (SQCQP) 中所有的 $(Q_1,\ldots,Q_k)$。如果某些约束不存在，则将其设置为 $\texttt{[ ]}$。参数 $\texttt{pars}$ 是可选的；但是，指定其中的一些参数可以提升求解器的性能和解的质量。  
</div>

<p style="line-height: 1;"></p>

```ruby
function Out = SNSQP(n,s,Q0,q0,Qi,qi,ci,ineqA,ineqb,eqA,eqb,lb,ub,pars)
% --------------------------------------------------------------------------------------------------
% This code aims at solving the sparse QCQP in the form of
%
%         min  (1/2)(x'{Q_0}x)+q_0'x  
%         s.t. (1/2)x'*Qi{i}*x+qi(:,i)'*x+ci(i)<=0, i = 1,...,k 
%                                 ineqA*x-ineqb<=0 
%                                      eqA*x-eqb=0 
%                                        lb<=x<=ub 
%                                       ||x||_0<=s 
% where Qi = {Qi{1},...,Qi{k}}, Qi{i} \in R^{n-by-n}, qi \in R^{n-by-k},  ci \in R^{k}
%       ineqA \in R^{m1-by-n},  ineqb \in R^{m1} 
%       eqA   \in R^{m2-by-n},  eqb   \in R^{m2}
%       s << n
% --------------------------------------------------------------------------------------------------           
% Inputs:
%     n:      Dimension of the solution x                                                 (REQUIRED)
%     s:      Sparsity level of x, an integer between 1 and n-1                           (REQUIRED)
%     Q0:     The quadratic objective matrix in R^{n-by-n}                                (REQUIRED)        
%     q0:     The quadratic objective vector in R^n                                       (REQUIRED)
%     Qi:     The quadratic constraint matrix                                             (OPTIONAL) 
%             MUST be a cell array or [], each entry is a matrix in R^{n-by-n}           
%     qi:     The quadratic constraint vector. MUST be a matrix in R^{n-by-k} or []       (OPTIONAL)           
%     ci:     The quadratic constraint constant in R, must be a vector or []              (OPTIONAL)
%     ineqA:  The linear inequality constraint matrix in R^{m1-by-n}   or []              (OPTIONAL)
%     ineqb:  The linear inequality constraint vector in R^{m1}        or []              (OPTIONAL)
%     eqA:    The linear equality constraint matrix in R^{m2-by-n}     or []              (OPTIONAL)
%     eqb:    The linear equality constraint vector in R^{m2}          or []              (OPTIONAL)
%     lb:     The lower bound of x                                                        (OPTIONAL)
%     ub:     The upper bound of x                                                        (OPTIONAL)
%             NOTE: 0 must be in [lb ub]
%     pars:   Parameters are all OPTIONAL
%             pars.x0       -- Initial point of x                                     (default zeros(n,1))
%             pars.dualquad -- Initial point of mu for quadratic constraints          (default zeros(k,1))
%             pars.dualineq -- Initial point of dual variable for linear inequalities (default zeros(m1,1))
%             pars.dualeq   -- Initial point of dual variable for linear equalities   (default zeros(m2,1))
%             pars.dualbd   -- Initial point of nu  for bound/box constraints         (default zeros(n,1))
%             pars.tau      -- A positive scalar                                      (default 1)
%                              NOTE: Tuning a proper tau may yield better solutions     
%             pars.itlser   -- Maximum number of line search                          (default 5)
%             pars.itmax    -- Maximum number of iterations                           (default 10000)
%             pars.show     -- Results shown at each iteration if pars.show=1         (default 1)
%                              Results not shown at each iteration if pars.show=0
%             pars.tol      -- Tolerance of the halting condition                     (default 1e-6)
% --------------------------------------------------------------------------------------------------
% Outputs:
%     Out.sol:           The sparse solution x
%     Out.sparsity:      Sparsity level of Out.sol
%     Out.error:         Error used to terminate this solver
%     Out.time           CPU time
%     Out.iter:          Number of iterations
%     Out.obj:           Objective function value at Out.sol
% --------------------------------------------------------------------------------------------------
% Send your comments and suggestions to <<< slzhou2021@163.com >>> 
% WARNING: Accuracy may not be guaranteed!!!!!  
% --------------------------------------------------------------------------------------------------
```
