using LinearAlgebra
using DelimitedFiles
using Plots
include("costfunc.jl")
include("gradDesc.jl")

x = 1:1:100

f(x) = 5*x .+ 7 
y = f(x) + randn(size(x,1)).*100
# plot data
plot(x,y, seriestype=:scatter)

#add bias column to X matrix
X = [ones(size(x,1)) x];
#initialize weights
theta = rand([0, 1], 2, 1);
iterations = 100;
lr = 0.01;

#compute initial loss
initialCost = costfunc(X, y, theta)

#run gradient descent
opt_theta, costHist = gradDesc(X,y,theta,lr,iterations)