using LinearAlgebra
using DelimitedFiles
using Plots
using Statistics

# create some data to classify
function f(x,a,b)
    a .* x[:,1] .+ x[:,2].*b;
end

function f_circle(x_c,m_c)
    m_c[:,1] .*x_c[:,1].^2 .+ m_c[:,2] .* x_c[:,2].^2 .+ m_c[:,3]
end

function sig(yPred)
    1 ./ (1 .+exp.(-1*yPred));
end

function cost(x,y,theta)
    mean(-y .* log.(sig(x*theta)) .-(1 .-y) .*log.(1 .-sig(x*theta)));
end

function grad(x,y,theta)
    1/size(y,1) .*(x' *(sig(x*theta) .- y));
end

function cost_grad_reg(x,y,theta,lambda)
    cost_reg = mean(-y .* log.(sig(x*theta)) .- (1 .- y) .*log.(1 .- sig(x*theta))) + lambda ./(2*size(x,1)) * sum(theta[2:end].^2);
    # grad for bias term
    grad1 = mean((sig(x*theta) .- y) .* x[:,1],dims=1)
    #grad for rest of theta
    grad_rest = mean((sig(x*theta) .- y) .* x[:,2:end],dims=1)' .+ lambda / size(x,1) .* theta[2:end]
    grad_full = vcat(grad1,grad_rest)
    return (cost_reg, grad_full)
end


a = -1;
b= 0.2;

num_data_points = 51;
points_min = -0.5
points_max = 1

m_c = Array{Float32}([0.3,0.5,-0.15])';

x_c = ones(num_data_points,2);
x_c[:,1] = range(points_min,points_max,length=num_data_points);
x_c[:,2] = range(points_min,points_max,length=num_data_points);

X = x_c[:,1]' .* ones(size(x_c,1));
Y =  ones(size(x_c,1))' .* x_c[:,2];
Z_c = ones(size(X));
i = j = 1
for i = 1:size(X,1)
    for j = 1:size(X,2)
        Z_c[i,j] = f_circle([X[i,j], Y[i,j]]',m_c)[1]
    end
end

x = range(points_min,points_max,length=num_data_points);
x = hcat(x,ones(length(x)));
descLine = f(x,a,b);
descLineY = zeros(0);

for i = 1:size(x,1)
    append!(descLineY,f(x[i,:]',a,b))
end

randomPoints = rand(size(x,1));
randomPoints .-= minimum(randomPoints);
randomPoints ./= maximum(randomPoints)
randomPoints = (randomPoints .+ points_min) .* points_max;

classes = copy(randomPoints);
classes .= 0;
classes_c = copy(classes);
i=1;
for i=1:size(randomPoints,1)
    yFunc = f(x[i,:]',a,b)[1]
    x_c_temp = hcat(x[i,1],randomPoints[i])
    y_c = f_circle(x_c_temp,m_c)[1]
    if randomPoints[i] > yFunc
        classes[i]=1;
    end
    if y_c < 0
        classes_c[i]=0
    else
        classes_c[i]=1
    end
end

boolIdx_c = classes_c .==1;

p = contour(x_c[:,1] ,x_c[:,2] ,Z_c,levels=[0]) # second axis showing is colorbar
plot!(x[boolIdx_c,1],randomPoints[boolIdx_c],seriestype=:scatter,show=true)
plot!(x[.!boolIdx_c,1],randomPoints[.!boolIdx_c],seriestype=:scatter,show=true)


boolIdx = classes .==1;
plot(x[:,1],descLineY,show=true)
plot!(x[boolIdx,1],randomPoints[boolIdx],seriestype=:scatter,show=true)
plot!(x[.!boolIdx,1],randomPoints[.!boolIdx],seriestype=:scatter,show=true)

# create polynomial features for non-linear descLine

deg = 3
x_poly = hcat(ones(size(x_c,1)),copy(x_c))
num = 1
for d = 1:deg
    for v = 0:d
        x_poly=hcat(x_poly, x_c[:,1] .^(d-v) .*x_c[:,2] .^v)
        num+=1
    end
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

plot(range(1,stop=epochs,step=1),losses,show=false)

prediction = round.(sig(x*theta));
accuracy = sum(prediction.==classes)/size(classes,1);

println("Accuracy for linear training data is $(accuracy)")

theta_c =rand(size(x_poly,2));
losses_c = ones(0);
lambda = 1e-2; # regularization term

for e in range(1,stop=epochs,step=1)
    (c,g) = cost_grad_reg(x_poly,classes_c,theta_c,lambda);
    append!(losses_c,c);
    theta_c .-= lr .* g[:];
end


plot(range(1,stop=epochs,step=1),losses_c,show=true)

prediction_c = round.(sig(x_poly*theta_c));
accuracy_c = sum(prediction_c.==classes_c)/size(classes_c,1);

println("Accuracy for circular training data is $(accuracy_c)")