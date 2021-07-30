using LinearAlgebra
using DelimitedFiles
using Plots
using Statistics

a = -1;
b= 1.0;

# create some data to classify
function f(x,a,b)
    a * x[1] .+ x[2].*b;
end
x = Array{Float32}(0:0.1:1);
x = hcat(x,ones(length(x)));
descLine = f(x,a,b);
descLineY = zeros(0);

for i = 1:size(x,1)
    append!(descLineY,f(x[i,:],a,b))
end

radomPoints = rand(size(x,1),2);

classes = copy(radomPoints[:,2]);
classes .= 0;
i=1;
for i=1:size(radomPoints,1)
    xTemp = hcat(radomPoints[i,1],1);
    yTemp = radomPoints[i,2];
    yFunc = f(xTemp,a,b);
    if yTemp > yFunc
        classes[i]=1;
    end
end

boolIdx = classes .==1;


plot(x[:,1],descLineY,show=true);
plot!(radomPoints[boolIdx,1],radomPoints[boolIdx,2],seriestype=:scatter);
plot!(radomPoints[.!boolIdx,1],radomPoints[.!boolIdx,2],seriestype=:scatter);

function sig(yPred)
    1 ./ (1 .+exp.(-1*yPred));
end

function cost(x,y,theta)
    m = size(x,1);
    mean(-y .* log.(sig(x*theta)) .-(1 .-y) .*log.(1 .-sig(x*theta)));
end

function grad(x,y,theta)
    1/size(y,1) .*(x' *(sig(x*theta) .- y));
end

theta = rand(2);

epochs = 1000;
lr=1e-1;
losses = ones(0);
for e in range(1,stop=epochs,step=1)
    loss = cost(x,classes,theta);
    append!(losses,loss);
    grads = grad(x,classes,theta);
    theta = theta .- lr .* grads 
end

plot(range(1,stop=epochs,step=1),losses,show=true)