#' fit a predictive model with robust boosting algorithm
#'
#' Fit a predictive model with robust boosting algorithm. For loss functions in the CC-family (concave-convex), apply composite optimization by conjugation operator (COCO), where optimization is conducted by functional descent boosting algorithm in xgboost. Models include the generalized linear models.

#' @param x input matrix, of dimension nobs x nvars; each row is an observation vector. Can accept \code{dgCMatrix}
#' @param y response variable. Quantitative for \code{dfun="greg:squarederror"}, \code{dfun="count:poisson"} (non-negative counts) or \code{dfun="reg:gamma"} (positive). For \code{dfun="binary:logitraw" or "binary:hinge"}, \code{y} should be a factor with two levels
#' @param weights vector of nobs with non-negative weights 
#' @param cfun concave component of CC-family, can be \code{"hacve", "acave", "bcave", "ccave"}, 
#' \code{"dcave", "ecave", "gcave", "hcave"}. See Table 2 in https://arxiv.org/pdf/2010.02848.pdf
#' @param s tuning parameter of \code{cfun}. \code{s > 0} and can be equal to 0 for \code{cfun="tcave"}. If \code{s} is too close to 0 for    \code{cfun="acave", "bcave", "ccave"}, the calculated weights can become 0 for all observations, thus crash the program
#' @param delta a small positive number provided by user only if \code{cfun="gcave"} and \code{0 < s <1}
#' @param dfun type of convex component in the CC-family, the second C, or convex down, that's where the name \code{dfun} comes from. It is the same as \code{objective} in the \code{xgboost} package.
#'   \itemize{
#'     \item \code{reg:squarederror} Regression with squared loss.
#'     \item \code{binary:logitraw} logistic regression for binary classification, predict linear predictor, not probabilies.
#'     \item \code{binary:hinge} hinge loss for binary classification. This makes predictions of -1 or 1, rather than   producing probabilities.
#'     \item \code{multi:softprob} softmax loss function for multiclass problems. The result contains predicted probabilities of each data point in each class, say p_k, k=0, ..., nclass-1. Note, \code{label} is coded as in [0, ..., nclass-1]. The loss function cross-entropy for the i-th observation is computed as -log(p_k) with k=lable_i, i=1, ..., n.
#'     \item \code{count:poisson}: Poisson regression for count data, predict mean of poisson distribution.
#'     \item \code{reg:gamma}: gamma regression with log-link, predict mean of gamma distribution. The implementation in \code{xgboost} takes a parameterization in the exponential family:\cr
#' xgboost/src/src/metric/elementwise_metric.cu.\cr
#' In particularly, there is only one parameter psi and set to 1. The implementation of the COCO algorithm follows this parameterization. See Table 2.1, McCullagh and Nelder, Generalized linear models, Chapman & Hall, 1989, second edition.
#'     \item \code{reg:tweedie}: Tweedie regression with log-link. @seealso \code{tweedie_variance_power} in \pkg{xgboost} with range: (1,2). A value close to 2 is like a gamma distribution. A value close to 1 is like a Poisson distribution.
#'}
#' @param iter number of iteration in the COCO algorithm
#' @param nrounds boosting iterations within each COCO iteration
#' @param del convergency criteria in the COCO algorithm, no relation to \code{delta}
#' @param trace if \code{TRUE}, fitting progress is reported
#' @param ... other arguments passing to \code{xgboost} 
#' @importFrom stats predict
#' @importFrom xgboost xgboost
#' @importFrom mpath update_wt compute_wt loss2 loss2_ccsvm loss3
#' @return An object with S3 class \code{xgboost}. \item{weight_update}{weight in the last iteration of the COCO algorithm} \item{los}{\code{iter} of sum of loss value of the composite function \code{cfun(dfun)} in the COCO iterations. Note, \code{cfun} requires \code{dfun} non-negative in some cases. Thus some \code{dfun} needs attentions. For instance, with \code{dfun="reg:gamma"}, to compute \code{los}, take first individual gamma-nloglik - (1+log(min(y))) to make all values
#' non-negative. This is because gamma-nloglik=y/ypre + log(ypre) in the \code{xgboost}, where ypre is the mean prediction value, can be negative. It can be derived that for fixed \code{y}, the minimum value of gamma-nloglik is achived at ypre=y, or 1+log(y). Thus, among all \code{y} values, the minimum of gamma-nloglik is 1+log(min(y)).}
#' @author Zhu Wang\cr Maintainer: Zhu Wang \email{zhuwang@gmail.com}
#' @references Wang, Zhu (2021), \emph{Unified Robust Boosting}, arXiv eprint, \url{https://arxiv.org/abs/2101.07718}
#' @keywords regression classification
#' @export ccboost
#' @examples
#'\donttest{
#' # regression, logistic regression, hinge regression, Poisson regression
#' x <- matrix(rnorm(100*2),100,2)
#' g2 <- sample(c(0,1),100,replace=TRUE)
#' fit1 <- ccboost(x, g2, cfun="acave",s=0.5, dfun="reg:squarederror", trace=TRUE, 
#'                 verbose=0, max.depth=1, nrounds=50)
#' fit2 <- ccboost(x, g2, cfun="acave",s=0.5, dfun="binary:logitraw", trace=TRUE,  
#'                 verbose=0, max.depth=1, nrounds=50)
#' fit3 <- ccboost(x, g2, cfun="acave",s=0.5, dfun="binary:hinge", trace=TRUE,  
#'                 verbose=0, max.depth=1, nrounds=50)
#' fit4 <- ccboost(x, g2, cfun="acave",s=0.5, dfun="count:poisson", trace=TRUE,      
#'                 verbose=0, max.depth=1, nrounds=50)
#'
#' # Gamma regression
#' x <- matrix(rnorm(100*2),100,2)
#' g2 <- sample(rgamma(100, 1))
#' library("xgboost")
#' fit5 <- xgboost(x, g2, objective="reg:gamma", max.depth=1, nrounds=50)
#' fit6 <- ccboost(x, g2, cfun="acave",s=5, dfun="reg:gamma", trace=TRUE, 
#'                 verbose=0, max.depth=1, nrounds=50)
#' plot(predict(fit5, x), predict(fit6, x))
#' hist(fit6$weight_update)
#' plot(fit6$los)
#' summary(fit6$weight_update)
#' 
#' # Tweedie regression 
#' fit6t <- ccboost(x, g2, cfun="acave",s=5, dfun="reg:tweedie", trace=TRUE, 
#'                 verbose=0, max.depth=1, nrounds=50)
#' # Gamma vs Tweedie regression
#' hist(fit6$weight_update)
#' hist(fit6t$weight_update)
#' plot(predict(fit6, x), predict(fit6t, x))
#'
#' # multiclass classification in iris dataset:
#' lb <- as.numeric(iris$Species)-1
#' num_class <- 3
#' set.seed(11)
#' 
#' # xgboost
#' bst <- xgboost(data=as.matrix(iris[, -5]), label=lb,
#' max_depth=4, eta=0.5, nthread=2, nrounds=10, subsample=0.5,
#' objective="multi:softprob", num_class=num_class)
#' # predict for softmax returns num_class probability numbers per case:
#' pred <- predict(bst, as.matrix(iris[, -5]))
#' # reshape it to a num_class-columns matrix
#' pred <- matrix(pred, ncol=num_class, byrow=TRUE)
#' # convert the probabilities to softmax labels
#' pred_labels <- max.col(pred)-1
#' # classification error
#' sum(pred_labels!=lb)/length(lb)
#'
#' # ccboost
#' fit7 <- ccboost(x=as.matrix(iris[, -5]), y=lb, cfun="acave", s=50,
#'                 dfun="multi:softprob", trace=TRUE, verbose=0, 
#'                 max.depth=4, eta=0.5, nthread=2, nrounds=10, 
#'                 subsample=0.5, num_class=num_class)
#' pred7 <- predict(fit7, as.matrix(iris[, -5]))
#' pred7 <- matrix(pred7, ncol=num_class, byrow=TRUE)
#' # convert the probabilities to softmax labels
#' pred7_labels <- max.col(pred7) - 1
#' # classification error: 0!
#' sum(pred7_labels != lb)/length(lb)
#' table(pred_labels, pred7_labels)
#' hist(fit6$weight_update)
#' }
ccboost <- function(x, y, weights, cfun="ccave", s=1, delta=0.1, dfun="reg:squarederror", iter=10, nrounds=100, del=1e-10, trace=FALSE, ...){
  call <- match.call()
  if(!dfun %in% c("reg:squarederror", "binary:logitraw", "binary:hinge", "multi:softprob", "count:poisson", "reg:gamma", "reg:tweedie"))
    stop("dfun not implemented/applicable")
  if(dfun %in% c("reg:gamma") && any(y <= 0))
    stop("response variable y must be positive for dfun ", dfun)
  eval(parse(text="mpath:::check_s(cfun, s)"))
  if(dfun %in% c("binary:logitraw", "binary:hinge")){
    ynew <- eval(parse(text="mpath:::y2num(y)"))
    y <- eval(parse(text="mpath:::y2num4glm(y)"))
  }else 
    ynew <- y
  cfunval <- eval(parse(text="mpath:::cfun2num(cfun)"))
  #what if dfun is not defined, such as gamma? it is worth to updating mpath
  dfunval <- switch(dfun,
                    "reg:squarederror"=1,
                    "binary:logitraw"=5,
                    "binary:hinge"=6,
                    "count:poisson"=8,
                    "reg:gamma"=NULL,
                    "reg:tweedie"=NULL,
                    "multi:softprob"=NULL)
  d <- 10 
  k <- 1
  if(trace) {
    cat("\nrobust boosting ...\n")
  }
  los <- rep(NA, iter)
  n <- length(y)
  if(missing(weights)) weights <- rep(1, n)
  ylos <- weights #initial values
  if(dfun=="reg:gamma")
    min_nloglik <- 1+log(min(y)) #the minimum value of negative log-likelihood value for a fixed y
  while(d > del && k <= iter){
    if(k==1) weight_update <- weights else
    weight_update <- mpath::compute_wt(ylos, weights, cfunval, s, delta)
    RET <- xgboost(data=x, label=y, weight=weight_update, objective = dfun, nrounds=nrounds, ...)
    ypre <- predict(RET, x) #depends on objective, this is probability or response or linear predictor
    #update loss values
    if(dfun=="reg:squarederror"){
      ylos <- (ynew - ypre)^2/2
    }else if(dfun=="binary:logitraw"){
    #u <- 1/(1+exp(-ypre)) # for y in [0, 1]
    #ylos <- -y*log(u/(1-u)) - log(1-u)
    ylos <- log(1 + exp( - ynew * ypre)) # for y in [-1, 1], the results ylos should be the same
    }else if(dfun=="binary:hinge"){
      ylos <- pmax(0, 1- ynew * ypre)
    }else if(dfun=="multi:softprob"){
      num_class <- RET$params$num_class
      # reshape it to a num_class-columns matrix
      ypre <- matrix(ypre, ncol=num_class, byrow=TRUE)
      ylos <- rep(NA, n)
      for(i in 1:n)
        ylos[i] = - log(ypre[i, y[i]+1]) # label y is coded as in [0, num_class-1]
    }else if(dfun %in% c("count:poisson")){
      ylos <- loss3(ynew, mu=ypre, theta=1, weights, cfunval, family=3, s, delta)$z
    }else if(dfun %in% c("reg:gamma")){
      ylos <- y/ypre+log(ypre) #negative log-likelihood value with "parameter"=1 in xgboost
      ylos <- ylos - min_nloglik #to shift the values to non-negative
    }else if(dfun %in% c("reg:tweedie")){
        #extract tweedie_variance_power
        rho <- substring(names(RET$evaluation_log[2]), 23)
        rho <- as.numeric(rho)
        a <- y * exp((1-rho)*log(ypre))/(1-rho)
        b <-     exp((2-rho)*log(ypre))/(2-rho)
        ylos <- - a + b
    }
    los[k] <- sum(mpath::compute_g(ylos, cfunval, s, delta))
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
  RET$los <- los
  RET
}
