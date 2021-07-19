using LinearAlgebra
using DelimitedFiles
using Plots

wd = pwd()
if ~occursin(wd).("Basic_Gradient_Descent_Julia")
    cd("$(wd)/Basic_Gradient_Descent_Julia")
end

include("costfunc.jl")
include("gradDesc.jl")

#set up data to be fitted
weight = .5;
bias = 0.7;
noise = 0.05;
epochs = 1500;
lr = 0.03;


x = 0:.05:1
f(x,weight,bias) = weight*x .+ bias 
y = f(x,weight,bias) + randn(size(x,1)).*noise

#add bias column to X matrix
X = [ones(size(x,1)) x];
#initialize weights
theta = rand([0, 1], 2, 1);


#compute initial loss
initialCost = costfunc(X, y, theta)

#run gradient descent
opt_theta, costHist = gradDesc(X,y,theta,lr,epochs);

#plot cost over epochs
p2=plot(1:epochs,costHist,ylabel="Cost", xlabel = "Epochs", lw = 3, title = "Cost History")

# compute fitted line
y_predict = f(x, opt_theta[1], opt_theta[2]);

#plot fit
p1=plot(x,y,ylabel="y", xlabel = "x", seriestype=:scatter,label="Data")
plot!(x,y_predict,lw=5, seriestype=:line,label="Prediction")
plot(p1,p2,layout=(2,1))