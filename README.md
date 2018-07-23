# Clustering-by-Nonparametric-Bayes
This is my final project of lecture -- 'Bayesian Inference's
date: "2018/7/8"

## Nonparametric Bayesian model
There are many cases, which require a presuming parametric specifications of the probability distributions, such as observations from Normal distribution or Quening model with Poisson distribution, and so on. But this subjective assumption may cause misspecification of model, which will result in bad prediction of severe lack of accuracy. To avoid such cases, nonparametric methods are better, which can allow more flexible and robust specification. "Learning from data" can help reduce many risk.

Indeed, nonparametric bayesian model should also have a bayesian-version style, which should contain prior, posterior, complex likelihood with kernel due to its complex expression and clever sampling methods needing expensive computation.

In machine learning, clustering is one big topic. There are many clustering methods. Clustering problem has an unsupervised problem setting, which means there is no available labels for the data, people have to use the intrisic structure of data to assign data points into different clusters.

To figure out this kind of problem, one way is to use a standard clustering algorithm like K-means or Gaussian mixture modeling. Both methods are similar, because they regard distances between data points as an important criterion for clustering, they cluster all points in a certain area together. But there is a problem that both methods need a fixed number of clusters, which should be setted by people subjectively. Although this problem can be solved by cross-validation in some extent, we now try a method, which doesn't need the prefixed total number of clusters. By resorting to nonparametric bayes, we allow the number of clusters grow as more data comes in instead of fixing the number of cluster to be discoverd.

## CRP process
CRP process is a proper generative model without fixed number of clusters. Consider a problem that we need one way to assign friend to different tables. We describe this process in a rigorous way:

* A chinese restaurant has an infinite number of tables:
* first cumtomer sits at the first table
* m-th subsequent customer sits at a table drawn from the following distribution:
![](http://mathurl.com/render.cgi?%24%24P%28previously%5C%20occupied%5C%20table%5C%20i%7C%5Cmathcal%7BF%7D_%7Bm-1%7D%29%20%5Cpropto%20n_i%24%24%5Cnocache)
![](http://mathurl.com/render.cgi?%24%24P%28the%5C%20next%5C%20unoccupied%5C%20table%7C%5Cmathcal%7BF%7D_%7Bm-1%7D%29%20%5Cpropto%20%5Calpha%24%24%5Cnocache)
where ![](http://latex.codecogs.com/gif.latex?%24n_i%24) is the number of customer currently at table i and where ![](http://mathurl.com/render.cgi?%5Cmathcal%7BF%7D_%7Bm-1%7D%5Cnocache) denotes the state of the restaurant after m-1 customers have been seated.

This means, the (n+1)st person sits down at a new table with probability ![](http://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Calpha%7D%7Bn&plus;%5Calpha%7D), and at table k with probability ![](http://latex.codecogs.com/gif.latex?%5Cfrac%7Bn_k%7D%7Bn&plus;%5Calpha%7D), where ![](http://latex.codecogs.com/gif.latex?n_k) is the number of people currently sitting at table k, and ![](http://latex.codecogs.com/gif.latex?%5Calpha) is a dispersion hyperparameter.

And with the clustering assignments, we can further assume that for kth cluster there is parameter $\phi_k$, we assume that $\phi_k\sim G_0$. Thus the generative model for data $X$ is $X|\phi_k \sim F_k, \phi_k\sim G_0$. This model can generate data with overlaping clusters. In this case, K-means and Gaussian mixture are not good enough, because from scatter plot people cannot define a correct number of clusters, because points are not seperable in the plot. 

The R-code for CRP process is following:

```{r cars}
####Chinese Restaurant Process####
crprnd = function(alpha, n){
  
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

```
This function will generate n samples from CRP process. It will return total number of clusters K and individual class vector "class".
```{r}
crprnd(alpha = 1.0, n = 20)
```
## Generate Data & Prior Assumption & Create Sampler
In this section, we will try the nonparametric bayesian clustering in a simple case, where data are generated from a Normal distribution with share constant variance. And $X_i$ is a p-dimension vector. The generative procedure is:

![](http://mathurl.com/render.cgi?%5Cphi_%7Bk%7D%5Csim%5Cmathcal%7BN%7D_p%28%5Cmu_0%2C%5CSigma_0%29%2C%5C%20%5C%20z_i%5Csim%20CRP%28%5Calpha%29%2C%20%5C%20%5C%20X_i%20%7C%20%5Cphi_k%2Cz_i%3Dk%2C%5CSigma%5Csim%5Cmathcal%7BN%7D_p%28%5Cphi_k%2C%5CSigma%29%5Cnocache)

### Generate Data
Here is the code for generating experimental dataset, we will generate 100 data points with 5 latent clusters. And we set ![](http://mathurl.com/render.cgi?%5Cphi_%7Bk%7D%5Csim%5Cmathcal%7BN%7D_p%28%5Cmu_0%2C%5CSigma_0%29%2C%5C%20%5C%20z_i%5Csim%20CRP%28%5Calpha%29%2C%20%5C%20%5C%20X_i%20%7C%20%5Cphi_k%2Cz_i%3Dk%2C%5CSigma%5Csim%5Cmathcal%7BN%7D_p%28%5Cphi_k%2C%5CSigma%29%5Cnocache), the generation procedure and simple visual is following: 
```{r}
library(mvtnorm)
library(MASS)
library(bayesm)
library(ggplot2)
mu0 = c(4,7)
sigma0 = matrix(c(4,1,1,8),2,2,byrow=T)
set.seed(1237)
cluster = crprnd(alpha =1 , 100)
Z = cluster$class
K = cluster$K
mu = mvrnorm(n = K,mu = mu0, Sigma = sigma0)
sig = diag(rep(1,2))*0.5
generateData = function(nk, muk, sigmak){
    testdata = NULL
    for(i in 1:length(nk)){
        t = mvrnorm(nk[i], mu = muk[i,],Sigma = sigmak)
        testdata = rbind(testdata, t)
    }
    testdata = cbind(testdata, rep(names(nk),nk))
    return(testdata)
}

nk = table(Z)
testdata = generateData(nk, mu, sig)
testdata = apply(testdata,2,as.numeric)
testdata = data.frame(testdata)
g = ggplot(testdata , aes(x = X1, y = X2, color = factor(X3)))
g+  geom_point()
```
From this graph we can see that, such dataser has obvious overlaping clusters, and these clusters can be detected easily.

### Derivation of Sampler
With this generative model, we can derive the posterior distribution of parameters with interest in. In clustering task, we want the distribution ![](http://latex.codecogs.com/gif.latex?%24z_i%7CX_i%2C%5CSigma%2C%5Cmu_0%2C%5CSigma_0%2C%20%5C%20%5C%20i%20%3D%201%2C...%2CN%2C%5C%20z_i%20%5Cin%20%5Cmathbb%7BN%5E&plus;%7D%24), and to get this posterior, which doesn't has a explicit expression. So we need to resort to Gibbs sampling method, which also need us to derive the full conditional distribution or respective kernel part. Because for parameters z we can only derive its kernel part, so we can only use ***Hybrid Gibbs Sampling***.

**Algorithm of Hybrid Gibbs**:

 1. Initialization. We set ![](http://latex.codecogs.com/gif.latex?%24%24%5Cmu_0%20%3D%20%5Cbar%7BX%7D%2C%5CSigma_0%3D%5Cfrac%7B1%7D%7Bn%7DXX%5E%7BT%7D%24%24) and initialize   ![](http://latex.codecogs.com/gif.latex?%24z_%7Bi%7D%2C%5Cphi_%7Bk%7D%2Ci%5Cin%5C%7B1%2C..%2CN%5C%7D%2Ck%5Cin%5C%7B1%2C..%2CK%5C%7D%24) with CRP and  multivarite normal.

 2. For the s-th echo:
     
    For i from 1 to N:
    
    * We use a proposal distribution ![](http://latex.codecogs.com/gif.latex?%24z%5E*%5Csim%20%5Cmathcal%7BN%7D_%7BI%28j%5Cin%5B0.5%2C100%5D%29%7D%28z%5E%7B%28s%29%7D_%7Bi%7D%2C%203%29%24), which is due to we want cluster index is positive integer and 100 points at most have 100 clusters;
    
    * Sample $z_i$ by *Metroplis-Hasting*. Calculate the ratio ![](http://latex.codecogs.com/gif.latex?%5Cfrac%7Bf%28z%5E*%7C%5Ccdot%29%7D%7Bf%28z%5Es%7C%5Ccdot%29%7D), where  ![](http://latex.codecogs.com/gif.latex?%24%24z_i%7C%5Cphi_%7B%5C%7Bk%5C%7D%7D%2CX_%7B%5C%7Bi%5C%7D%7D%2Cz_%7B-i%7D%2C%5CSigma_0%2C%5Cmu_0%5Cpropto%20%5Csum_%7Bk%3D1%7D%5E%7BK%7Dn_kp%28x_i%7C%5Cphi_k%291_%7B%5C%7Bz_i%3Dk%5C%7D%7D%20&plus;%20%5Calpha%20f%28x_i%7C%5Cmu_0%2C%5CSigma_0%291_%7B%5C%7Bz_i%3DK&plus;1%5C%7D%7D%24%24) where ![](http://latex.codecogs.com/gif.latex?%24n_k%24) is the number of points in cluster k.
    
    For k from 1 to K:
    * Sample ![](http://latex.codecogs.com/gif.latex?%24%5Cphi_k%24) from the full conditional distribution. According to the posterior update, we have:
    ![](http://latex.codecogs.com/gif.latex?%24%24%5Cphi_k%7CX_%7Bi%7D%2CZ_%7Bi%7D%2C%5Cmu_0%2C%5CSigma_0%20%5Csim%20%5Cmathcal%7BN%7D_p%5Cbig%28%28%5CSigma_0%5E%7B-1%7D&plus;%5CSigma%5E%7B-1%7D%29%5E%7B-1%7D%28%5CSigma%5E%7B-1%7D%5Cbar%7BX%7D_k&plus;%5CSigma_0%5Cmu_0%29%2C%28%5CSigma_0%5E%7B-1%7D&plus;%5CSigma%5E%7B-1%7D%29%5E%7B-1%7D%5Cbig%29%24%24)
    
 3. If s>C, C is the echo when this sampler is stationary, we will collect samples of ![](http://latex.codecogs.com/gif.latex?%24%28z_1%2C...%2Cz_N%29%24).
 
 4. Stop sampling until s reach S.

This model is also named Dirichlet Process Gaussian Mixture Model(DPGMM) or Infinite Gaussian Mixture Model.

Code for calculating the loglikelihood ratio and Hybird Gibbs are following:
```
#####likelihood function for Gibbs#####
likeli = function(zi,x_i,n_vec,alpha,Kplus,mu, mu0, sigma0, sig){
    p = length(x_i)
    if(zi > Kplus){
        return(alpha*dmvnorm(x_i, mean = mu0, sigma = sigma0+sig))
    }
    else return(n_vec[zi]*dmvnorm(x_i,mean = mu[zi,], sigma = sig))
}

mu0 = colMeans(testdata[,1:2])
sigma0 = cov(testdata[,1:2])
plot(testdata[,1],testdata[,2], pch = 20)

####### Start the Simulation!!!!######
alpha = 1; 
R = 10000; N = 100; p = 2
sig = diag(rep(1,p))*0.5
mean_vec = matrix(0)
cluster = rep(0, N)
####Initialization: clusters, means#####
cluster = crprnd(alpha, N)
K = cluster$K
mu = mvrnorm(100, mu0, sigma0)
Z = cluster$class
mean_vec1 = tapply(testdata[,1],Z, mean)
mean_vec2 = tapply(testdata[,2],Z, mean)
mean_vec = cbind(mean_vec1, mean_vec2)
n_vec = table(Z); K = length(n_vec)
X = testdata[,1:2]

######updating by Hybrid Gibbs#######
sigma_update = solve(solve(sig) + solve(sigma0))
invsigma0 = solve(sigma0)
Z_post = NULL

for(r in 1:R){
    if(r %% 50 == 0)
        print(paste0("this process is still alive!! Be patient! Echo = ",r))
    for( i in 1:N){
        s = 1; S = 5
        z = NULL
        z.s = Z[i]
        
        for(s in 1:S){
            ####choose proper proposal distribution####
            z.star = rtrun(2, 3, 0.5, 100)
            z.star = round(z.star)
            ###Computer the rejection probability
            log.r = log(likeli(z.star,X[i,],n_vec,alpha,K,mu, mu0, sigma0, sig)) - 
                log(likeli(z.s,X[i,],n_vec,alpha,K,mu, mu0, sigma0, sig))
            
            if (log(runif(1))< log.r) {z.s = z.star}
            z = c(z,z.s)
        }
        Z[i] = z[S]
    }
    ####update vec mu and vec n#######
    for(k in 1:K){
        mu_update = sigma_update%*%(solve(sig)%*%mean_vec[k,] + invsigma0 %*% mu0)
        mu[k,] =  mvrnorm(1 ,mu = mu_update, Sigma = sigma_update)
    }
    
    n_vec = table(Z) ; K = length(n_vec)
    mean_vec1 = tapply(testdata[,2],Z, mean)
    mean_vec2 = tapply(testdata[,3],Z, mean)
    mean_vec = cbind(mean_vec1,mean_vec2)
    if(r%%100 == 0)
        paste0('the process is still alive, r=',r)
    if(r > 7000){
        Z_post = rbind(Z_post,Z)
    }
}

####get the posterior mode####
getcluster = function(x){
    return(which.max(table(x)))
}

####mean####
classes = round(apply(Z_post,2,mean))

finaldata = cbind(X,classes) 
####Visualization####
ggplot(finaldata, aes(x=X1, y=X2, shape = factor(classes),color=factor(classes))) + geom_point()
```
(After about 15mins..) The final visualization is: 

![](nonparabayes.png)

From this graph we can see that this method can catch some overlaping clusters and allow more flexibility. Although we have only 4 clusters here, but each cluster seem reasonable. A better performence may be got by other additional tools, such as sample $z_i$ with a certain gap and keep sampler working for a longer time.

## Kmeans method from Frequentists
Kmeans method is a clustering method based on distrances between data points, and the procedure is resorting to EM algorithm. In this procedure, people regard labels as missing data, and use EM algorithm to assign data points in the cluster which makes data has largest likelihood, and then update the new cluster parameters by sample means. Code is following:
```{r}
kmclass = kmeans(testdata, 5)
final = cbind(testdata[,1:2], kmclass$cluster)
names(final) = c('x1','x2','c')
g = ggplot(final, aes(x= x1, y= x2, color= factor(c)))
g+ geom_point()
```

## Conclusion
From the results of both methods, we found that nonparametric bayes can handle more flexible dataset, which has complex intrinsic structure or correlation. But Kmeans-like frequentists' methods mostly consider that under fixed groups how can people assign these data points properly. And clustering by nonparametric bayes don't need to choose the number of clusters, which will give a smart guess when people don't know anything about datasets in hand. And the model can be imporved with cluster-specified covariance matrix. Even more complex case is that we are interested in all parameters, we can create a bigger Gibbs sampler which includes every full conditional distributions, but the computation will be very expensive. Because i-th label's likelihood is based on current lablels of other data, there is not a trival parallel computation strategy for this problem.
