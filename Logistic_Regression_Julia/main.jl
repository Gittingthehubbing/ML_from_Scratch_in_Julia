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

radomPoints = rand(100,2);

classes = copy(radomPoints);
classes[:,2] .= 0;
i=1;
for i=1:size(radomPoints,1)
    xTemp = hcat(radomPoints[i,1],1);
    yTemp = radomPoints[i,2];
    yFunc = f(xTemp,a,b);
    if yTemp > yFunc
        classes[i,2]=1;
    end
end

boolIdx = classes[:,2].==1;


plot(x[:,1],descLineY,show=true)
plot!(radomPoints[boolIdx,1],radomPoints[boolIdx,2],seriestype=:scatter)
plot!(radomPoints[.!boolIdx,1],radomPoints[.!boolIdx,2],seriestype=:scatter)

function sig(x)
    1/(1 .+exp(-1*x));
end

function cost(x,y,theta)
    m = size(x,1);
    mean(-y * log(sig(theta'*x))-(1 .-y)*log(1-sig(theta' * x)));
end

function grad(x,y,theta)
    mean((sig(theta' *x) .- y)*x)
end

thetaInit = rand(2,1);

costInit = cost(x,classes, thetaInit)
gradInit = grad(x,classes,theta)