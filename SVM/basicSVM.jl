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


yR = -1
yT = -2:0.1:3
l = []
for y in yT
    lT = hinge_loss(y, yR)
    push!(l,lT)
end

yR = 1
l2 = []
for y in yT
    lT = hinge_loss(y, yR)
    push!(l2,lT)
end
plot!(yT,l2)

xVals, y_true = MLJ.make_blobs(200, 2,centers=2)

xMatrix =Tables.matrix(xVals)
yMatrix = convert(Vector{Int64},coerce(y_true,Continuous))
y_true[y_true .== 2] .= -1
yMatrix[yMatrix .== 2] .= -1

xDf = DataFrame(xVals)
fullDf = DataFrame(xVals)
fullDf.y_true = y_true

hinge_loss(ones(size(y_true)[1]).+3,coerce(y_true,Count))

w = randn(2)
b= randn(1)
losses = []

initial_loss = runSVM(xMatrix,w,b,yMatrix)

testGrad = gradient(runSVM,xMatrix,w,b,yMatrix)

losses, w, b= trainSVM(xMatrix, yMatrix,  3000, 0.01)
y_pred = svm(xMatrix,w,b)
trained_loss = hinge_loss(y_pred,yMatrix)

plot(1:size(losses)[1], losses, show=true, yaxis=:log)

labels_pred = ones(Int,size(y_pred)) .*-1
labels_pred[y_pred .> 1] .*= -1 
labels_pred[y_pred .< -1] .*= -1 

comp = [convert(Vector{Int64},yMatrix) labels_pred]

plot(xMatrix[:,1],xMatrix[:,2],c=convert(Vector{Int64},
    yMatrix),seriestype= :scatter,α=0.6,show=true, label="Real",marker = (:star,13))
plot!(
    xMatrix[:,1],xMatrix[:,2],c=labels_pred,
    seriestype= :scatter,marker = (:o,4),α=1, label="Predicted",show=true,
    legend=:topright)

