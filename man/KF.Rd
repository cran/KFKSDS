\name{KF}
\alias{KF}
\alias{KF.C}
\alias{KF.deriv}
\alias{KF.deriv.C}

\title{Kalman Filter for State Space Models}

\description{
These functions run the iterative equations of the Kalman filter 
for a state space model.
}

\usage{
KF(y, ss, convergence = c(0.001, length(y)), t0 = 1)
KF.C(y, ss, convergence = c(0.001, length(y)), t0 = 1)
KF.deriv(y, ss, xreg = NULL, convergence = c(0.001, length(y)), t0 = 1)
KF.deriv.C(y, ss, xreg = NULL, convergence = c(0.001, length(y)), 
  t0 = 1, return.all = FALSE)
}

\arguments{
\item{y}{a numeric time series or vector.}
\item{ss}{a list containing the matrices of the state space model.}
\item{xreg}{optional matrix or list of external regressors. See details.}
\item{convergence}{a numeric vector of length two to control and determine 
the convergence of the filter. See details below.}
\item{t0}{a numeric indicating the index of the first observation 
at which the contributions to the likelihood are addep up.}
\item{return.all}{logical. If \code{TRUE}, extended output containing elements 
to be used by \code{\link[KFKSDS]{KS.deriv}} is returned.}
}

\section{State space representation}{
The general univariate linear Gaussian state space model is defined 
as follows:
\deqn{y[t] = Za[t] + e[t], e[t] \sim N(0, H)}
\deqn{a[t+1] = Ta[t] + Rw[t], w[t] \sim N(0, V)}

for \eqn{t=1,\dots,n} and \eqn{a[1] \sim N(a0, P0)}.
\eqn{Z} is a matrix of dimension \eqn{1\times m}{1xm}; 
\eqn{H} is \eqn{1\times 1}{1x1};
\eqn{T} is \eqn{m\times m}{mxm};
\eqn{R} is \eqn{m\times r}{mxr};
\eqn{V} is \eqn{r\times r}{rxr}; 
\eqn{a0} is \eqn{m\times 1}{mx1} and
\eqn{P0} is \eqn{m\times m}{mxm},
where \eqn{m} is the dimension of the state vector \eqn{a} and 
\eqn{r} is the number of variance parameters in the state vector.

The Kalman filtering recursions for the model above are:

\emph{Prediction}
\deqn{a[t] = T a[t-1]}
\deqn{P[t] = T P[t-1] T' + R V R'} 
\deqn{v[t] = y[t] - Z a[t]}
\deqn{F[t] = Z P[t] Z' + H} 

\emph{Updating}
\deqn{K[t] = P[t] Z' F[t]^{-1}}
\deqn{a[t] = a[t] + K[t] v[t] }
\deqn{P[t] = P[t] - K[t] Z P[t]'}

for \eqn{t=2,\dots,n}, starting with \eqn{a[1]} and \eqn{P[1]} equal
to \code{a0} and \code{P0}. \eqn{v[t]} is the prediction error at observation 
in time \eqn{t} and \eqn{F[t]} is the variance of \eqn{v[t]}.
}

\details{
The implementation is a direct transcription of the iterative equations 
of the filter that are summarized below. Details can be found in the references given 
below and in many other textbooks.
The source code follows the notation used in Durbin and Koopman (2001).

The elements in the argument \code{ss} must be named in accordance with 
the notation given below for the state space representation.
For those models defined in the package \pkg{stsm},
a convenient way to create the argument \code{ss} is by means of the 
function \code{\link[stsm]{char2numeric}}.

The contributions to the likelihood function of the first observations 
may be omitted by choosing a value of \code{t0} greater than one.

The functions with \sQuote{\code{.deriv}} in the name compute the derivatives of some 
of the elements involved in the filter with respect to the parameters of the model.

The functions \code{KF} and \code{KF.deriv} are fully implemented 
in \R while \code{KF.deriv.C} calls to compiled C code.

Using \code{KF.deriv.C} with \code{return.all = FALSE} returns minimal output with 
the elements necessary to compute the derivative of the log-likelihood function.
Using \code{return.all = TRUE} further elements to be used in 
\code{\link[KFKSDS]{KS.deriv}} are returned.

Missing observations are allowed. If a missing value is observed after the 
filter has converged then all operations of the filter are run instead of using 
steady state values until convergence is detected again. 
%Therefore, the presence of missing values reduces the computational saving.

\emph{Parameters to control the convergence of the filter.}
In some models, the Kalman filter may converge to a steady state. Finding the 
explicit expression of the steady state values can be cumbersome in some models.
Alternatively, at each iteration of the filter it can be checked whether a steady state 
has been reached. For it, some control parameters can be defined in the argument \code{convergence}. 
It is considered that convergence was reached when the following is observed:
the change in the variance of the prediction error over the last \code{convergence[2]} 
consecutive iterations of the filter is below the tolerance value \code{convergence[1]}.
When the steady state is reached, the values from the last iteration are used 
in the remaining iteration of the filter. %and smoother. 
Thus, the number of matrix operations can be reduced substantially as pointed 
in Harvey (1989) Sec. 3.3.4.
%H89 pp.124 Sec. 3.3.4, convergence can be exploited to 
%reduce computations, matrix operations
%
%If value of the likelihood function happens to be infinite 
%(caused by a value of the error prediction error equal to zero), 
%it is set equal to \eqn{99999}.

\emph{External regressors.}
A matrix of external regressors can be passed in argument \code{xreg}.
If \code{xreg} is a matrix then it is assumed that the time series 
passed in argument \code{y} has been already adjusted for the effect 
of these regressors, that is, \eqn{y_t^{adj} = y_t - X \gamma}.
If \code{y} is the observed series, then \code{xreg} should be a list containing 
the following elements: \code{xreg}, the matrix of regressors; and 
\code{coefs}, the vector of coefficients, \eqn{\gamma}, related to the regressors.
The coefficients must be placed in \code{coefs} in the same order 
as the corresponding vectors are arranged by columns in \code{xreg}.

The number of rows of the matrix of regressors must be equal to the length 
of the series \code{y}. 

Column names are necessary for \code{KF.deriv} and 
are optional for \code{KF.deriv.C}.
}

\value{
A list containing the following elements:
\item{v}{prediction error.}
\item{f}{variance of the prediction error.}
\item{K}{Kalman gain.}
\item{L}{auxiliar matrices to be used by the smoother.}
\item{a.pred}{one step ahead prediction of the state vector.}
\item{P.pred}{covariance matrix of \code{a.pred}.}
\item{a.upd}{update of \code{a.pred} after the observation at time \eqn{t} 
that is predicted in \code{a.pred} enters in the recursive filter.}
\item{P.upd}{update of \code{P.pred}.}
\item{convit}{the iteration at which the filter converged. If convergence 
was not observed it is set to \code{NULL}.}
\item{mll}{value of the negative of the log-likelihood function.}

The function \code{KF.C} is a simplified and faster version that 
records and returns only the value of negative of the log-likelihood function.
It is suited to be passed as argument to \code{\link[stats]{optim}} in the 
\pkg{stats} package.

The functions that evaluate the derivatives include in the output the derivatem 
terms: \code{da.pred}, \code{dP.pred}, \code{da.upd}, \code{dP.upd}, \code{dv}, \code{df}, 
\code{dvof} (derivative of quotient \code{v} over \code{f}), \code{dK} and \code{dL}.

\code{KF.deriv.C} does not return \code{a.upd} and \code{P.upd} and their 
derivative terms \code{da.upd} and \code{dP.upd}. 
If \code{return.all = TRUE}, this function returns: \code{dvof}, 
\code{dL}, \code{da.pred}, \code{dP.pred}, which are the derivative terms necessary
to evaluate the gradient of the log-likelihood function. By default they are not returned,
\code{return.all = FALSE}. They are in any case computed, the operations that are omitted 
in the latter case is the arrangement of the output from the call to compiled C code into 
matrices of the proper dimension containing the data in the right order.

Derivatives of the likelihood function are implemented in package \pkg{stsm}. 
Although the Kalman filter provides information to evaluate the likelihood function, 
it is not its primary objective. That's why the derivatives of the likelihood are 
currently part of the package \pkg{stsm}, which is specific to likelihood methods 
in structural time series models.
}

\references{
Durbin, J. and Koopman, S. J. (2001).
\emph{Time Series Analysis by State Space Methods}. 
Oxford University Press.

Harvey, A. C. (1989).
\emph{Forecasting, Structural Time Series Models and the Kalman Filter}. 
Cambridge University Press.
}

\seealso{
\code{\link[stsm]{char2numeric}} in package \pkg{stsm}.
}

\examples{
# local level plus seasonal model with arbitrary parameter values
# for the 'JohnsonJohnson' time series
m <- stsm::stsm.model(model = "llm+seas", y = JohnsonJohnson, 
  pars = c("var1" = 2, "var2" = 15, "var3" = 30))
ss <- stsm::char2numeric(m)

# run the Kalman filter
kf <- KF(m@y, ss)
plot(kf$a.upd[,1:2], main = "filtered state vector")

# 'KF.C' is a faster version that returns only the 
# value of the negative of the likelihood function
kfc <- KF.C(m@y, ss)
all.equal(kf$mll, kfc)

# compute also derivative terms used below
kfd <- KF.deriv(m@y, ss)
all.equal(kfc, kfd$mll)
kfdc <- KF.deriv.C(m@y, ss, return.all = TRUE)
all.equal(kf$mll, kfdc$mll)

# as expected the versions that use compiled C code
# are faster that the versions written fully in R
# e.g. not including derivatives
\dontrun{
system.time(for(i in seq_len(10)) kf <- KF(m@y, ss))
system.time(for(i in seq_len(10)) kfc <- KF.C(m@y, ss))
# e.g. including derivatives
system.time(for(i in seq_len(10)) kfd <- KF.deriv(m@y, ss))
system.time(for(i in seq_len(10)) kfdc <- KF.deriv.C(m@y, ss, return.all = TRUE))
}

# compare analytical and numerical derivatives
# they give same results up to a tolerance error
fcn <- function(x, model, type = c("v", "f"))
{
  m <- stsm::set.pars(model, x)
  ss <- stsm::char2numeric(m)
  kf <- KF(m@y, ss)
  switch(type, "v" = sum(kf$v), "f" = sum(kf$f))
}

dv <- numDeriv::grad(func = fcn, x = m@pars, model = m, type = "v")
all.equal(dv, colSums(kfd$dv), check.attributes = FALSE)
all.equal(dv, colSums(kfdc$dv), check.attributes = FALSE)
df <- numDeriv::grad(func = fcn, x = m@pars, model = m, type = "f")
all.equal(df, colSums(kfd$df), check.attributes = FALSE)
all.equal(df, colSums(kfdc$df), check.attributes = FALSE)

# compare timings in version written in R with numDeriv::grad
# no calls to compiled C code in either case
# looking at these timings, using analytical derivatives is
# expected to be useful in optimization algorithms
\dontrun{
system.time(for (i in seq_len(10)) 
  numdv <- numDeriv::grad(func = fcn, x = m@pars, model = m, type = "v"))
system.time(for(i in seq_len(10)) kfdv <- colSums(KF.deriv(m@y, ss)$dv))
}

# compare timings when convergence is not checked with the case 
# when steady state values are used after convergence is observed
# computation time is reduced substantially
\dontrun{
n <- length(m@y)
system.time(for(i in seq_len(20)) a <- KF.deriv(m@y, ss, convergence = c(0.001, n)))
system.time(for(i in seq_len(20)) b <- KF.deriv(m@y, ss, convergence = c(0.001, 10)))
# the results are the same up to a tolerance error
all.equal(colSums(a$dv), colSums(b$dv))
}
}

\keyword{ts, model}
