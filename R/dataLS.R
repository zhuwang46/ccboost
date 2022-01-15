#' generate random data for classification as in Long and Servedio (2010)

#' @param ntr number of training data
#' @param ntu number of tuning data, default is the same as \code{ntr}
#' @param nte number of test data
#' @param percon proportion of contamination, must between 0 and 1. If \code{percon > 0}, the labels of the corresponding percenrage of response variable in the training and tuning data are flipped.
#' @return a list with elements xtr, xtu, xte, ytr, ytu, yte for predictors of disjoint training, tuning and test data, and response variable -1/1 of training, tuning and test data.
#' @author Zhu Wang\cr Maintainer: Zhu Wang \email{zhuwang@gmail.com}
#' @references P. Long and R. Servedio (2010), \emph{Random classification noise defeats all convex potential boosters}, \emph{Machine Learning Journal}, 78(3), 287--304.
#' @keywords classification
#' @export dataLS
#' @examples
#' dat <- dataLS(ntr=100, nte=100, percon=0)
dataLS <- function(ntr, ntu=ntr, nte, percon){
    n <- ntr+ntu+nte
                                        #The data from this distribution can be perfectly classified by sign(\sum_i x_i), i.e., sign(apply(x, 1, sum))==y
    x <- matrix(NA, nrow=n, ncol=21)
                                        #lable y is chosen to be -1 or +1 with equal probability
    y <- sample(c(-1, 1), n, replace=T,prob=c(0.5,0.5))
                                        #generate large margin, pullers and penaliziers or numbers 1/2/3 with probabilty 0.25/0.25/0.5, respectively
    z <- sample(c(1, 2, 3), n, replace=T,prob=c(0.25,0.25,0.5))
    for(i in 1:n){
        if(z[i]==1)
            x[i,]=y[i]
        else if(z[i]==2){
            x[i,1:11]=y[i]
            x[i,12:21]=-y[i]
        }
        else if(z[i]==3){
### randomly choose 5 out of the first 11 features
            t1 <- sample(1:11)
### randomly choose 6 out of the last 10 features
            t2 <- sample(12:21)
            x[i, c(t1[1:5], t2[1:6])] <- y[i]
### remaining 
            x[i, -c(t1[1:5], t2[1:6])] <- -y[i]
        }
    }
### contaminated data
    xtr <- x[1:ntr,]
    ytr <- y[1:ntr]
    xtu <- x[(1+ntr):(ntr+ntu),]
    ytu <- y[(1+ntr):(ntr+ntu)]
    xte <- x[(1+ntr+ntu):n,]
    yte <- y[(1+ntr+ntu):n]
    con <- sample(ntr) 
    contu <- sample(ntu) 
    if(percon > 0){
        j <- con[1:(percon*ntr)]
        jtu <- contu[1:(percon*ntu)]
        ytr[j] <- -ytr[j]
        ytu[jtu] <- -ytu[jtu]
    }
    RET <- list(xtr=xtr, ytr=ytr, xtu=xtu, ytu=ytu, xte=xte, yte=yte)
}
