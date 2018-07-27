function [partition, m, K, partition_bin] = crprnd(alpha, n)

%--------------------------------------------------------------------------
  % EXAMPLE
% alpha = 3; n= 100;
% [partition, m, K, partition_bin] = crprnd(alpha, n);
%--------------------------------------------------------------------------
  
#Check parameter
crprnd = function(alpha, n){

 if(alpha<=0){
  error('Parameter alpha must be a positive scalar')
 }

 m = rep(0,n);
 partition = rep(0, n);
 partition_bin = matrix(rep(0,n*n),n,n);

 # Initialization
 partition[1] = 1
 partition_bin[1,1] = 1
 m[1] = 1
 K = 1
 # Iterations
 for(i in 2:n){
  # Compute the probability of joining an existing cluster or a new one
  proba = c(m[1:K], alpha)/(alpha+i-1)
  # Sample from a discrete distribution w.p. proba
  u = runif(1)
  partition[i] = which(u<=cumsum(proba))[1]
  partition_bin[i,partition[i]] = 1
  # Increment the size of the cluster
  m[partition[i]] = m[partition[i]] + 1
  # Increment the number of clusters if new
  if(sum(m>0) > K) K = K + 1;     
 }
 return(list(class = partition,class_size = m, K = K, class_decrp = partition_bin))
}

dpstick= function(alpha, K){
 w0 = rbeta(K, 1, alpha);
 stick = c(1,cumprod(1-w0))[1:K]
 w = stick * w0
 return(w)
}


###Pitman-Yor CRP
pycrp = function(alpha,sigma,n){
  m = rep(0,n);
  partition = rep(0, n);
  partition_bin = matrix(rep(0,n*n),n,n);
  
  # Initialization
  partition[1] = 1
  partition_bin[1,1] = 1
  m[1] = 1
  K = 1
  # Iterations
  for(i in 2:n){
    # Compute the probability of joining an existing cluster or a new one
    proba = c(m[1:K]-sigma, alpha+sigma*K)/(alpha+i-1)
    # Sample from a discrete distribution w.p. proba
    u = runif(1)
    partition[i] = which(u<=cumsum(proba))[1]
    partition_bin[i,partition[i]] = 1
    # Increment the size of the cluster
    m[partition[i]] = m[partition[i]] + 1
    # Increment the number of clusters if new
    if(sum(m>0) > K) K = K + 1;     
  }
  return(list(class = partition,class_size = m, K = K, class_decrp = partition_bin))
}


##
# Generate some fake data with some uniform random means
##
generateFakeData = function( num.vars=3, n=100, num.clusters=5, seed=NULL ) {
  if(is.null(seed)){
    set.seed(runif(1,0,100))
  } else {
    set.seed(seed)
  }
  data <- data.frame(matrix(NA, nrow=n, ncol=num.vars+1))
  
  mu <- NULL
  for(m in 1:num.vars){
    mu <- cbind(mu,rnorm(num.clusters, runif(1,-10,15), 5))
  }
  
  for (i in 1:n) {
    cluster <- sample(1:num.clusters, 1)
    data[i, 1] <- cluster
    for(j in 1:num.vars){
      data[i, j+1] <- rnorm(1, mu[cluster,j], 1)
    }
  }
  
  data$X1 <- factor(data$X1)
  var.names <- paste("VAR",seq(1,ncol(data)-1), sep="")
  names(data) <- c("cluster",var.names)
  
  return(data)
}


##
# Set up a procedure to calculate the cluster means using squared distance
##
dirichletClusters <- function(orig.data, disp.param = NULL, max.iter = 100, tolerance = .001)
{
  n <- nrow( orig.data )
  
  data <- as.matrix( orig.data )
  pick.clusters <- rep(1, n)
  k <- 1
  
  mu <- matrix( apply(data,2,mean), nrow=1, ncol=ncol(data) )
  
  is.converged <- FALSE
  iteration <- 0
  
  ss.old <- Inf
  ss.curr <- Inf
  
  while ( !is.converged & iteration < max.iter ) { # Iterate until convergence
    iteration <- iteration + 1
    
    for( i in 1:n ) { # Iterate over each observation and measure the distance each observation' from its mean center's squared distance from its mean
      distances <- rep(NA, k)
      
      for( j in 1:k ){
        distances[j] <- sum( (data[i, ] - mu[j, ])^2 ) # Distance formula.
      }
      
      if( min(distances) > disp.param ) { # If the dispersion parameter is still less than the minimum distance then create a new cluster
        k < - k + 1
        pick.clusters[i] <- k
        mu <- rbind(mu, data[i, ])
      } else {
        pick.clusters[i] <- which(distances == min(distances))
      }
      
    }
    
