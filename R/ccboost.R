#' fit a predictive model with robust boosting algorithm
#'
#' Fit a predictive model with robust boosting algorithm. For loss functions in the CC-family (concave-convex), apply composite optimization by conjugation operator (COCO), where optimization is conducted by functional descent boosting algorithm. Models include the generalized linear models.

#' @param x input matrix, of dimension nobs x nvars; each row is an observation vector. Can accept \code{dgCMatrix}
#' @param y response variable. Quantitative for \code{dfun="gaussian"}, or \code{dfun="poisson"} (non-negative counts). For \code{dfun="binomial" or "hinge"}, \code{y} should be a factor with two levels
#' @param weights vector of nobs with non-negative weights 
#' @param cfun concave component of CC-family, can be \code{"hacve", "acave", "bcave", "ccave"}, 
#' \code{"dcave", "ecave", "gcave", "hcave"}
#' @param s tuning parameter of \code{cfun}. \code{s > 0} and can be equal to 0 for \code{cfun="tcave"}. If \code{s} is too close to 0 for    \code{cfun="acave", "bcave", "ccave"}, the calculated weights can become 0 for all observations, thus crash the program
#' @param delta a small positive number provided by user only if       \code{cfun="gcave"} and \code{0 < s <1}
#' @param dfun type of convex component in the CC-family, can be \code{"gaussian", "binomial"}, \code{"hinge", "poisson"}
#' @param iter number of iteration in the COCO algorithm
#' @param nrounds boosting iterations
#' @param del convergency criteria in the COCO algorithm
#' @param trace if \code{TRUE}, fitting progress is reported
#' @param ... other arguments passing to \code{xgboost} 
#' @importFrom stats predict
#' @importFrom xgboost xgboost
#' @importFrom mpath update_wt compute_wt loss2 loss2_ccsvm loss3
#' @return An object with S3 class \code{xgboost}. \item{weight_update}{weight in the last iteration of the COCO algorithm}
#' @author Zhu Wang\cr Maintainer: Zhu Wang \email{zhuwang@gmail.com}
#' @references Wang, Zhu (2021), \emph{Unified Robust Boosting}, arXiv eprint, \url{https://arxiv.org/abs/2101.07718}
#' @keywords regression classification
#' @export ccboost
#' @examples
#'\donttest{
#' x <- matrix(rnorm(100*2),100,2)
#' g2 <- sample(c(0,1),100,replace=TRUE)
#' fit1 <- ccboost(x, g2, cfun="acave",s=0.5, dfun="gaussian", trace=TRUE, 
#'                 verbose=0, max.depth=1, nrounds=50)
#' fit2 <- ccboost(x, g2, cfun="acave",s=0.5, dfun="binomial", trace=TRUE,  
#'                 verbose=0, max.depth=1, nrounds=50)
#' fit3 <- ccboost(x, g2, cfun="acave",s=0.5, dfun="hinge", trace=TRUE,  
#'                 verbose=0, max.depth=1, nrounds=50)
#' fit4 <- ccboost(x, g2, cfun="acave",s=0.5, dfun="poisson", trace=TRUE,      
#'                 verbose=0, max.depth=1, nrounds=50)
#'}
ccboost <- function(x, y, weights, cfun="ccave", s=1, delta=0.1, dfun="gaussian", iter=10, nrounds=100, del=1e-10, trace=FALSE, ...){
    call <- match.call()
    if(!dfun %in% c("gaussian", "binomial", "hinge", "poisson"))
    stop("dfun not implemented yet")
        #if(dfun %in% c("binomial", "hinge"))
        #if(!all(names(table(y)) %in% c(1, -1)))
        #    stop("response variable must be 1/-1 for dfun ", dfun)
    eval(parse(text="mpath:::check_s(cfun, s)"))
     theta <- 0 ### not used yet
     if(dfun=="gaussian") family <- 1 else
         if(dfun=="binomial") family <- 2 else
     if(dfun=="poisson"){
         family <- 3
     }else if(dfun=="negbin"){
         stop("not implemented for dfun='negbin'")
         if(missing(theta)) stop("theta has to be provided for family='negbin'")
         family <- 4
     }
        bsttype <- switch(dfun,
                          "gaussian"="reg:squarederror",
                          "binomial"="binary:logitraw",
                          "hinge"="binary:hinge",
                          "poisson"="count:poisson",
                          "cox"="survival:cox")
    #if(dfun %in% c("hinge")) 
    #    stop("check how the hinge loss is defined with y in 0/1 for xgboost")
    if(dfun %in% c("binomial", "hinge")){
         ynew <- eval(parse(text="mpath:::y2num(y)"))
         y <- eval(parse(text="mpath:::y2num4glm(y)"))
    }else 
        ynew <- y
    cfunval <- eval(parse(text="mpath:::cfun2num(cfun)"))
    dfunval <- eval(parse(text="mpath:::dfun2num(dfun)"))
    d <- 10 
    k <- 1
    if(trace) {
        cat("\nrobust boosting ...\n")
    }
    los <- rep(NA, iter)
    n <- length(y)
    if(missing(weights)) weights <- rep(1, n)
    weight_update <- weights
    while(d > del && k <= iter){
        #param <- list(booster = "gblinear", objective = bsttype, nthread = 2, ...)
### probably don't need lambda, alpha above
        if(dfun=="binomial") RET <- xgboost(data=x, label=y, weight=weight_update, objective = bsttype, eval_metric="logloss", nrounds=nrounds, ...) else
        RET <- xgboost(data=x, label=y, weight=weight_update, objective = bsttype, nrounds=nrounds, ...)
        ypre <- predict(RET, x) #depends on objective, this is probability or response or linear predictor
        if(dfun %in% c("gaussian", "binomial", "hinge")){
        weight_update <- update_wt(ynew, ypre, weights, cfunval, s, dfunval)
        if(dfun=="hinge") los[k] <- loss2_ccsvm(ynew, ypre, weights, cfunval, dfun="C-classification", s, eps=0, delta) else
        los[k] <- loss2(ynew, ypre, weights, cfunval, dfunval, s, delta)
        }else{
        #RET <- xgboost(param, data=cbind(x, y), label=y, weight=weight_update, nrounds = 200, objective = bsttype, eta = 0.8, updater = 'coord_descent', feature_selector = 'thrifty', top_k = 1, ...)
        tmp1 <- loss3(ynew, mu=ypre, theta, weights, cfunval, family, s, delta)
        weight_update <- compute_wt(tmp1$z, weights, cfunval, s, delta)
        los[k] <- sum(tmp1$tmp)
        }
        if(trace) cat("\niteration", k, "nrounds", nrounds, ": los[k]", los[k], "d=", d, "\n") 
	if(k > 1){
	    d <- abs((los[k-1]-los[k]))/los[k-1]
		   if(los[k] > los[k-1])
	       nrounds <- nrounds + 100	    
    }
        k <- k + 1
    }
    RET$x <- x
    RET$y <- y
    RET$call <- call
    RET$weight_update <- weight_update
    RET
}
