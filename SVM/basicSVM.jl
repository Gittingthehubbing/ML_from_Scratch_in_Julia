using MLJ
using Zygote
using Plots
using DataFrames

function hinge_loss(y_pred, y_true)
    loss = 1 .- y_pred .*y_true
end

svm(x, w, b) = sum(w .* x) .+ b

x, y_true = MLJ.make_blobs(100, 2)
y_true