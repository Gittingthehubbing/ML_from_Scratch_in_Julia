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
    prediction = sig(x*theta)
    cost_reg = mean(-y .* log.(prediction) .- (1 .- y) .*log.(1 .- prediction)) + lambda ./(2*size(x,1)) * sum(theta[2:end].^2);
    # grad for bias term
    grad1 = mean((sig(x*theta) .- y) .* x[:,1],dims=1)
    #grad for rest of theta
    grad_rest = mean((sig(x*theta) .- y) .* x[:,2:end],dims=1)' .+ lambda / size(x,1) .* theta[2:end]
    grad_full = vcat(grad1,grad_rest)
    return (cost_reg, grad_full)
end

function main() #somewhat lazy way to avoid global scope issue in for loops

a = -1;
b= 0.2;

num_data_points = 50;
points_min = -1.5
points_max = 2
normalize_non_lin_data = true

plot_dir = "plots"
mkpath(plot_dir)

m_c = Array{Float32}([1.2,1.1,-0.8])';

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

randomPoints = rand(points_min:0.1:points_max,size(x,1));

classes = rand(0:1:1,size(randomPoints,1));
classes_c = copy(classes);
i=1;
for i=1:size(randomPoints,1)
    yFunc = f(x[i,:]',a,b)[1]
    x_c_temp = hcat(x[i,1],randomPoints[i])
    y_c = f_circle(x_c_temp,m_c)[1]
    if randomPoints[i] > yFunc
        classes[i]=1;
    else
        classes[i]=0;
    end
    if y_c < 0
        classes_c[i]=1
    else
        classes_c[i]=0
    end
end

# create polynomial features for non-linear descLine

deg = 6
#x_poly = hcat(ones(size(x_c,1)),copy(x_c))
x_poly = ones(size(x_c,1));
#num = 1 #this does not work
num = ones(1) # avoids issue of global scope
for d = 1:deg
    for v = 0:d
        x_poly = hcat(x_poly, x_c[:,1] .^(d-v) .*x_c[:,2] .^v);
        num[1]+=1
    end
end

#Normalisation
if normalize_non_lin_data
    colMeanList = zeros(0);
    colStdList = zeros(0);
    for col in range(1,size(x_poly,2),step=1)
        xMeanC = mean(x_poly[:,col])
        xStd = std(x_poly[:,col])
        append!(colMeanList,xMeanC)
        append!(colStdList,xStd)
        x_poly[:,col] .-= xMeanC
        x_poly[:,col] ./= (xStd+1e-8) #in case of zeros
    end
end
theta = rand(2);

epochs = 10;
lr=1e-2;
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


boolIdx = classes .==1;
plot(x[:,1],descLineY);
plot!(x[boolIdx,1],randomPoints[boolIdx],seriestype=:scatter);
plot!(x[.!boolIdx,1],randomPoints[.!boolIdx],seriestype=:scatter,show=true)
savefig("$(plot_dir)/Linear-Target.png")

theta_c =rand(size(x_poly,2))./10;
losses_c = ones(0);
lambda = 1e-1; # regularization term

for e in range(1,stop=epochs,step=1)
    (c,g) = cost_grad_reg(x_poly,classes_c,theta_c,lambda);
    append!(losses_c,c);
    theta_c .-= lr .* g[:];
end


plot(range(1,stop=epochs,step=1),losses_c,yaxis=:log,show=false)
savefig("$(plot_dir)/Non-linear-losses.png")

prediction_c = round.(sig(x_poly*theta_c));
accuracy_c = sum(prediction_c.==classes_c)/size(classes_c,1);

println("Accuracy for circular training data is $(accuracy_c)")


boolIdx_c = classes_c .==1;

p = contour(x_c[:,1] ,x_c[:,2] ,Z_c,levels=[0]); # second axis showing is colorbar
plot!(x_c[boolIdx_c,1],randomPoints[boolIdx_c],seriestype=:scatter);
plot!(x_c[.!boolIdx_c,1],randomPoints[.!boolIdx_c],seriestype=:scatter,show=true)
savefig("$(plot_dir)/Non-Linear-Target.png")

boolHits = prediction_c .== 1;
p2 = contour(x_c[:,1] ,x_c[:,2] ,Z_c,levels=[0]); 
plot!(x_c[boolHits,1],randomPoints[boolHits],seriestype=:scatter,label ="Predicted outside",lw=4)
plot!(x_c[.!boolHits,1],randomPoints[.!boolHits],seriestype=:scatter,label ="Predicted inside",lw=3)
savefig("$(plot_dir)/Non-Linear-hits.png")

println("All done")
end

main()