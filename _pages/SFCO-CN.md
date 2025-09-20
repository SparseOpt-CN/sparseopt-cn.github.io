---
layout: archive
title: ""   
permalink: /SFCO-CN/
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

## 阶梯函数约束优化
---

<p style="line-height: 1;"></p>
\begin{equation}
\min_{\mathbf{x}\in\mathbb{R}^{K}} ~~  f(\mathbf{x}),~~~~ \mbox{s.t.}~~ \parallel\mathbf{G}(\mathbf{x})\parallel_0^+\leq s,~\mathbf{x}\in \Omega  \tag{SFCO}
\end{equation}

<div style="text-align:justify;">
其中， 矩阵 $\mathbf{G}(\mathbf{x})\in\mathbb{R}^{M \times N}$ 的第 $(i,j)$ 个元素为 $G_{ij}(\mathbf{x})$，函数 $f:\mathbb{R}^{K}\rightarrow \mathbb{R}$ 和 $G_{ij}:\mathbb{R}^{K}\rightarrow \mathbb{R}$ 连续可微，最好二次连续可微，正整数 $s\ll n$，集合 $\Omega\subseteq\mathbb{R}^{K}$ 闭凸。度量 $\|\mathbf{Z}\|_0^+$ 计算矩阵 $\mathbf{Z}\in\mathbb{R}^{M \times N}$ 中含有正元素的列的个数，即
  \begin{equation}\|\mathbf{Z}\|_0^+= \mathrm{step}\Big(\max_{i=1,\ldots,M} Z_{i1}\Big)+\cdots+\mathrm{step}\Big(\max_{i=1,\ldots,M} Z_{iN}\Big)\nonumber\end{equation}
这里，$\mathrm{step}$ 是阶梯（又称 0/1 损失）函数，定义为：$\mathrm{step}(t)=1$ 当 $t>0$；否则 $\mathrm{step}(t)=0$。当 $M=1$，矩阵 $\mathbf{Z}$ 退化成向量 $\mathbf{z}\in\mathbb{R}^{N}$，如果令 $\mathbf{z}_+=$ $(\max\{0,z_1\}$ $\ldots$ $\max\{0,z_N\})^\top$ 以及零范数 $\parallel\mathbf{z}\parallel_0$ 计算 $\mathbf{z}$ 中非零元个数，则有  
  \begin{equation*}\|\mathbf{z}\|_0^+= \mathrm{step}(z_1)+\cdots+\mathrm{step}(z_N)=\|\mathbf{z}_+\|_0\end{equation*}
目前， 集合 $\Omega$ 可以是以下集合之一：    
</div> 

- 球: $\lbrace\mathbf{x}: \parallel\mathbf{x}\parallel^2\leq r\rbrace$，其中 $r>0$: 
- 半空间: $\lbrace\mathbf{x}: \mathbf{a}^T\mathbf{x}\leq b\rbrace$，其中 $\mathbf{a}\in\mathbb{R}^{K}$ 和 $b\in\mathbb{R}$
- 超平面: $\lbrace\mathbf{x}: \mathbf{A} \mathbf{x}=  \mathbf{b}\rbrace$，其中 $\mathbf{A}\in\mathbb{R}^{S\times K}$ 和 $ \mathbf{b}\in\mathbb{R}^{S}$
- 盒子:  $\lbrace\mathbf{x}: l\leq x_i \leq u, i=1,\ldots,K\rbrace$，其中 $l \leq u$ 分别可以取值 $-\infty$ 和 $+\infty$。因此，当 $l=-\infty$ 和 $u=+\infty$， $\Omega$ 为无约束；当 $l=0$ 和 $u=+\infty$， $\Omega$ 为非负约束
 
<!-- ## <span style="color:#8C8C8C"> The solver and its demonstration </span> -->

---
<div style="text-align:justify;"> 
程序包 - <a style="font-size: 16px; font-weight: bold;color:#006DB0" href=" " target="_blank">SFCOpack-Matlab</a> 或 <a style="font-size: 16px; font-weight: bold;color:#006DB0" href=" " target="_blank">SFCOpack-Python</a>（点击下载）提供了 1 个求解器，其核心算法来自以下文章：
</div>

> <b style="font-size:14px;color:#777777">SNSCO</b> - <span style="font-size: 14px"> S Zhou, L Pan, N Xiu,  and G  Li, A 0/1 constrained optimization solving sample average approximation for chance constrained programming, MOR, 2024. </span>

---
<div style="text-align:justify;">  
求解器 $\texttt{SNSCO}$ 核心算法属于二阶方法，需要用到目标函数 $f(\mathbf{x})$ 和 约束函数 $\mathbf{G}(\mathbf{x})$ 的函数值、梯度和海瑟矩阵。基于 Matlab 语言（基于 Python语言，可进行类似定义）下面用一个恢复问题作为示例，展示如何为该求解器定义这些内容。 恢复问题的目标函数为 $f(\mathbf{x})=0.5\parallel\mathbf{B}\mathbf{x}-\mathbf{d}\parallel^2$ 和约束函数为 $\mathbf{G}_{ij}(\mathbf{x})= \lbrace\mathbf{A}_{:(j-1)M+i},\mathbf{x}\rbrace-C_{ij}$。下面两段 MATLAB 代码分别定义了$f$ 和 $\mathbf{G}$ 的函数值、梯度和海瑟矩阵。例如，函数句柄 $\texttt{FuncfRecovery}$ 的输入中，$\texttt{x}$ 为变量，（$\texttt{B}$，$\texttt{d}$，$\texttt{BtB}$）为函数 $f(\mathbf{x})$ 中涉及的数据。在调用函数 $\texttt{FuncfRecovery}$ 时，这些数据需要用户自己定义。 
</div>
<p style="line-height: 1;"></p>

```ruby
function [objef, gradf, hessf] = FuncfRecovery(x,B,d,BtB)
    % This code provides information for an objective function
    %     f(x) = 0.5*||Bx-d||^2  
    % x is the variable 
    % (B,d,BtB) are data and need to be input
    % B\in R^{m by n}, d\in R^{m by 1}, and BtB = B'*B
    
    Bxd   = B*x-d;
    objef = norm(Bxd)^2/2;  % objective
    if  nargout>1
        gradf = (Bxd'*B)';  % gradient
        hessf = BtB;        % Hessian
    end
    clear B d BtB
end
```

```ruby
function [G,gradG,gradGW, hessGW] = FuncGRecovery(x,W,Ind,A,C,K,M,N) 
    % This code provides information for function G(x): R^K -> R^{M x N}
    % (x,W,Ind) are variables
    % (A,C,K,M,N) are data and parameters, and need to be input 
    % For each i=1,...,M and j=1,...,N:
    %           G_{ij}(x) = <A_{:,(j-1)*M+i}, x>^2 - C_{ij} 
    % where A_{:,(j-1)*M+i} is the ((j-1)*M+i)th column of A
    % A \in R^{K by M*N} and C \in R^{M by N} 
    
    G0  = x'*A;
    G   = reshape(G0,M,N);
    G   = G.^2-C;                                         % function
    if  nargout > 1
        if  isempty(Ind) 
            gradG   = [];
            gradGW  = zeros(K,1);                         % gradient
            hessGW  = zeros(K,1);                         % Hessian
        else 
            AInd    = A(:,Ind);    
            gradG   = AInd.*G0(Ind); 
            WInd    = W(Ind);
            gradGW  = gradG*reshape(WInd,length(Ind),1);  % gradient
            hessGW  = AInd*diag(WInd)*AInd.';             % Hessian     
        end
    end  
    clear A C K M N
end
```

<div style="text-align:justify;">  
定义完函数 $f$ 和 $\mathbf{G}$ 后，用户需要选择约束集 $\Omega$。目前，$\Omega$ 允许取 {'$\texttt{Ball}$', '$\texttt{Box}$', '$\texttt{Halfspace}$', '$\texttt{Hyperplane}$'} 中的一个。而每一个集合会涉及相应的参数。例如，盒子约束需要设置上下界，当下界取 $-\infty$ 和 上界取 $+\infty$ 时， 盒子约束变为无约束；当下界取 $0$ 和 上界取 $+\infty$ 时，盒子约束变为非负约束。对于其他约束，相关参数详见上面的模型介绍。选择完约束集 $\Omega$ 后，就可以调用 $\texttt{SNSCO}$ 来求解问题。下面的 Matlab 代码展示了求解恢复问题的过程。
</div>

<p style="line-height: 1;"></p>

```ruby
% demon recovery problems
clc; close all; clear all; addpath(genpath(pwd));

K     = 10; 
M     = 10; 
N     = 100;
alpha = 0.05;
s     = ceil(alpha*N);

sets = {'Ball','Box','Halfspace','Hyperplane'};
test = 1;  % Omega = {x|norm(x) <= r}  if test = 1
           % Omega = [lb,ub]^n         if test = 2    
           % Omega = {x|a'*x <= b}     if test = 3 
           % Omega = {x|Ax = b}        if test = 4 
switch sets{test}
    case 'Ball'
        input1  = 2;
        input2  = [];
        xopt    = randn(K,1);
        xopt    = input1/norm(xopt)*xopt;
    case 'Box'
        input1  = -2;
        input2  = 2;
        xopt    = input1 + (input2-input1)*rand(K,1); 
    case 'Halfspace'
        xopt    = rand(K,1);
        input1  = randn(K,1);
        input2  = sum(input1.*xopt)+rand(1); 
    case 'Hyperplane'
        xopt    = randn(K,1);
        input1  = randn(ceil(0.5*K),K);
        input2  = input1*xopt; 
end

% Generate data and define f and G
B           = randn(ceil(0.25*K),K)/sqrt(K);
d           = B*xopt;
BtB         = B'*B;
xi          = randn(K,M,N);
T           = randperm(N,s);
Mat         = rand(M,N);
Delta       = (Mat>=0.5) .* rand(M,N);
Delta(:,T)  = (Mat(:,T)<1/3).*rand(M,nnz(T))-(Mat(:,T)>=2/3).*rand(M,nnz(T)); 
A           = reshape(xi,K,M*N);
C           = (squeeze(sum(xi .* xopt, 1))).^2 + Delta; 
Funcf      = @(x)FuncfRecovery(x,B,d,BtB);            % f(x)    = 0.5||Bx-d||^2
FuncG      = @(x,W,J)FuncGRecovery(x,W,J,A,C,K,M,N);  % G(x)_ij = <A_ij,x>^2-C_ij

% set parameters and call the solver
if  alpha  > 0.01
    pars.tau0 = 0.5+0.5*(test>4);
else
    pars.tau0 = 0.01;
    pars.thd  = 1e-1*(test==4)+1e-2*(test~=4);
end
out  = SNSCO(K,M,N,s,Funcf,FuncG,sets{test},input1,input2,pars);
fprintf(' Relerr:    %7.3e \n', norm(out.x-xopt)/norm(xopt));  
```

<div style="text-align:justify;">
Matlab 版求解器 $\texttt{SNSCO}$ 的输入与输出（Python 版的输入与输出类似）说明如下，其中输入 ($\texttt{K}$, $\texttt{M}$, $\texttt{N}$, $\texttt{s}$, $\texttt{Funcf}$, $\texttt{FuncG}$, $\texttt{FeaSet}$, $\texttt{input1}$, $\texttt{input2}$) 为必需项。$\texttt{pars}$ 中的参数为可选项，但设置某些参数可能会提升求解器的性能和解的质量。 需要注意的是，$\texttt{FeaSet}$ 只能取 {'$\texttt{Ball}$', '$\texttt{Box}$', '$\texttt{Halfspace}$', '$\texttt{Hyperplane}$'} 中的一个。对于每一个集合，求解器设置了两个输入 $\texttt{input1}$ 和 $\texttt{input2}$。当不需要某个输入时，可以设置为空 $\texttt{[ ]}$。例如，当 $\texttt{FeaSet}$='$\texttt{Ball}$' 时， 可设置 $\texttt{input1}$=2 和 $\texttt{input2}$=$\texttt{[ ]}$，表示半径为 2 的球约束。当 $\texttt{FeaSet}$='$\texttt{Box}$' 时， 可设置 $\texttt{input1}$=0 和 $\texttt{input2}$ = $\texttt{Inf}$，表示非负约束。
</div>

<p style="line-height: 1;"></p>

```ruby
function out = SNSCO(K,M,N,s,Funcf,FuncG,FeaSet,input1,input2,pars)
% This solver solves 0/1 constrained optimization in the following form:
%
%         min_{x\in\R^K} f(x),  s.t. \| G(x) \|^+_0<=s, x\in Omega 
%
% where 
%      f(x) : \R^K --> \R
%      G(x) : \R^K --> \R^{M-by-N}
%      s << N 
%      \|Z\|^+_0 counts the number of columns with positive maximal values
%      Omega is a closed and convex set
% -------------------------------------------------------------------------
% Inputs:
%   K     : Dimnesion of variable x                              (REQUIRED)
%   M     : Row number of G(x)                                   (REQUIRED)
%   N     : Column number of G(x)                                (REQUIRED)
%   s     : An integer in [1,N), typical choice ceil(0.01*N)     (REQUIRED)
%   Funcf : Function handle of f(x)                              (REQUIRED)
%   FuncG : Function handle of G(x)                              (REQUIRED)
%   FeaSet: Feasible set for x, must be one of:                  (REQUIRED)
%          'Box'             [lb,ub]^K
%          'Ball'            {x|norm(x) <= r} 
%          'Halfspace'       {x|a'*x <= b}
%          'Hyperplane'      {x|Ax = b}
%           Default:         R^K
%   input1: A parameter related to FeasSet                       (REQUIRED)
%   input2: A parameter related to FeasSet                       (REQUIRED)
%   pars  : All parameters are OPTIONAL  
%           pars.x0    -- Initial point (default:ones(K,1)) 
%           pars.tau0  -- A vector of a number of  \tau0       (default  1)
%                         e.g.,pars.tau0=logspace(log10(.5),log10(1.75),20) 
%           pars.tol   -- Tolerance of halting condition (default 1e-6*M*N)
%           pars.maxit -- Maximum number of iterations       (default 2000) 
%           pars.disp  -- Show results or not at each step      (default 1)
% -------------------------------------------------------------------------
% Outputs:
%     out.x:      Solution x
%     out.obj:    Objective function value f(x)
%     out.G:      Function value of G(x) 
%     out.time:   CPU time
%     out.iter:   Number of iterations 
%     out.error:  Error
%     out.Error:  Error of every iteration
% -------------------------------------------------------------------------
% Written by Shenglong Zhou on 30/4/2024 based on the algorithm proposed in
%     Shenglong Zhou, Lili Pan, Naihua Xiu, and Geoffrey Ye Li, 
%     0/1 constrained optimization solving sample average approximation 
%     for chance constrained programming, Math Oper Res, 2024    	
% Send your comments and suggestions to <<< slzhou2021@163.com >>> 
% WARNING: Accuracy may not be guaranteed!!!!!  
% -------------------------------------------------------------------------
```
