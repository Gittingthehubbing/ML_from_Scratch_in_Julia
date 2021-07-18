using LinearAlgebra
using DelimitedFiles
using Plots
include("costfunc.jl")
include("gradDesc.jl")

#set up data to be fitted
weight = .5;
bias = 0.7;
noise = 2;
epochs = 3;
lr = 0.001;


x = 1:1:100
f(x,weight,bias) = weight*x .+ bias 
y = f(x,weight,bias) + randn(size(x,1)).*noise
# plot data
plot(x,y, seriestype=:scatter)

#add bias column to X matrix
X = [ones(size(x,1)) x];
#initialize weights
theta = rand([0, 1], 2, 1);


#compute initial loss
initialCost = costfunc(X, y, theta)

#run gradient descent
opt_theta, costHist = gradDesc(X,y,theta,lr,epochs);

#plot cost over epochs
plot(1:epochs,costHist)

# compute fitted line
y_predict = f(x, opt_theta[1], opt_theta[2]);

#plot fit
plot(x,y_predict, seriestype=:scatter)