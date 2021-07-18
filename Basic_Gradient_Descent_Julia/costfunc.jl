function costfunc(X,y,theta) #last expression will be returned if no return statement
    m = size(y,1)
    J = 1/(2*m) * sum((X*theta-y).^2)

end