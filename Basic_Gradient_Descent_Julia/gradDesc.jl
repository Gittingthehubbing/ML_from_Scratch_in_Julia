function gradDesc(X,y,theta, lr, epochs)
    m=size(y,1)
    J_history = zeros(epochs, 1)

    for iter = 1:epochs

        gradJ = lr/m * X'*(X*theta - y)
        theta = theta - gradJ
        Jloop = costfunc(X,y,theta)
        J_history[iter] = Jloop

    end
    return theta, J_history
end