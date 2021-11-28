using MLJ
using Zygote
using Plots
using DataFrames
using Statistics
using Tables

function hinge_loss(y_pred, y_true)
    lossVal2 = mean((max.(0, 1 .- y_pred .* y_true)).^2)
end
svm(x, w, b) = x*w .+ b
runSVM(x,a,b,y) = hinge_loss(svm(x,a,b),y)

function trainSVM(xMatrix, yMatrix, epochs,lr)
    w = randn(2)
    b= randn(1)
    losses = []
    for e in 1:epochs
        loss, grads = withgradient(runSVM,xMatrix,w,b,yMatrix)
        append!(losses,loss)
        w .-= lr .* grads[2]
        b -= lr * grads[3]
    end
    return losses, w, b
end


function makeMarginMap(xVals,yVals,yMatrix,w,b,nMeshpoints = 30)
        
    x1Min = minimum(xVals)
    x2Min = minimum(yVals)
    x1Max = maximum(xVals)
    x2Max = maximum(yVals)
    
    x1Range = LinRange(x1Min,x1Max,nMeshpoints)
    x2Range = LinRange(x2Min,x2Max,nMeshpoints*2)

    cMesh = [svm([x1 x2],w,b)[1] for x1= x1Range,x2=x2Range]

    cMesh[cMesh .>1] .=4
    cMesh[0 .< cMesh .<1] .=3
    cMesh[0 .> cMesh .> -1] .=2
    cMesh[cMesh .<-1] .=1

    heatmap(x1Range,x2Range,cMesh')
    plot!(xVals, yVals, c=yMatrix, seriestype=:scatter,legend=false)
end



yT = -2:0.1:3
l = [hinge_loss(y, -1) for y in yT]

plot(yT,l)

l2 = [hinge_loss(y, 1) for y in yT]

plot!(yT,l2)

nDatapoints = 200

useBlobs = true


if useBlobs
    xVals, y_true = MLJ.make_blobs(nDatapoints, 2,centers=2)

    xMatrix =Tables.matrix(xVals)
    xMatrix .-= mean(xMatrix)
    xMatrix ./= std(xMatrix)
    yMatrix = convert(Vector{Int64},coerce(y_true,Continuous))

    yMatrix[yMatrix .== 2] .= -1
    plot(xMatrix[:,1], xMatrix[:,2], c=yMatrix, seriestype=:scatter)
else
    randFunc(x) = x .+ maximum(x) .*randn(size(x))
    xVals = LinRange(-.8,.8,nDatapoints)
    yVals = randFunc(xVals)
    
    xMatrix = [xVals yVals]
    
    descLine(x) = 5 .*x .^5 .- 0.4
    yMatrix = [y < descLine(x) ? 1 : -1 for (x,y) in zip(xVals,yVals)]
    plot(xMatrix[:,1], descLine(xMatrix[:,1]), label="Descision Line",linewidth=5)
    plot!(xMatrix[:,1], xMatrix[:,2], c=yMatrix, seriestype=:scatter)
end

w = randn(2)
b= randn(1)

makeMarginMap(xMatrix[:,1],xMatrix[:,2],yMatrix,w,b, 300)

losses = []

initial_out = svm(xMatrix,w,b)
initial_loss = runSVM(xMatrix,w,b,yMatrix)

losses, w, b= trainSVM(xMatrix, yMatrix,  3000, 0.01)

y_pred = svm(xMatrix,w,b)
trained_loss = hinge_loss(y_pred,yMatrix)

plot(1:size(losses)[1], losses, show=true, yaxis=:log)

labels_pred = ones(Int,size(y_pred)) .*-1
labels_pred[y_pred .> 0] .*= -1 

comp = [convert(Vector{Int64},yMatrix) labels_pred]
correctPreds = [
    x == y ? 1 : 0 for (x,y) in zip(labels_pred, yMatrix)
]
acc = sum(correctPreds)/length(correctPreds)
println("Accuracy is $(floor(Int,acc*100)) %")
makeMarginMap(xMatrix[:,1],xMatrix[:,2],yMatrix,w,b, 300)

