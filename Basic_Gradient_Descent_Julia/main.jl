using LinearAlgebra
using DelimitedFiles
using Plots
using Statistics

wd = pwd()
if ~occursin(wd).("Basic_Gradient_Descent_Julia")
    cd("$(wd)/Basic_Gradient_Descent_Julia")
end

include("costfunc.jl")
include("gradDesc.jl")

#set up data to be fitted
weight = .5;
bias = 0.7;
noise = 0.15;
epochs = 150;
lr = 0.00003;

normalise=true


x = Array{Float32}(1:0.05:10);
x_Before = copy(x);
f(x,weight,bias) = weight*x .+ bias ;
y = f(x,weight,bias) + randn(size(x,1)).*noise;

#Normalise features
if normalise
    x_Recovered = rand(Array{Float32}(minimum(x):0.1:maximum(x)),size(x,1))
    for col=1:size(x,2)
        println("col ",col)
        mean_x = mean(x[:,col],dims=1);
        std_x = std(x[:,col]);
        x[:,col] =  (x[:,col] .- mean_x)/std_x;
        x_Recovered[:,col] = (x[:,col].*std_x).+mean_x;
    end
    difference = x_Before .- x_Recovered;
    println("Diff is ",sum(difference))
end

#add bias column to X matrix
X = [ones(size(x,1)) x];
#initialize weights
theta = rand(0:0.01:1, 2, 1);


#compute initial loss
initialCost = costfunc(X, y, theta)

#run gradient descent
opt_theta, costHist = gradDesc(X,y,theta,lr,epochs);

#plot cost over epochs
p2=plot(
    1:epochs,costHist,
    ylabel="Cost", xlabel = "Epochs", 
    lw = 3, title = "Cost History", yaxis=:log)

# compute fitted line
y_predict = f(x_Before, opt_theta[1], opt_theta[2]);


#plot fit
p1=plot(x,y,ylabel="y", xlabel = "x", seriestype=:scatter,label="y_real")
plot!(x,y_predict,lw=5, seriestype=:line,label="Prediction")
plot(p1,p2,layout=(2,1))