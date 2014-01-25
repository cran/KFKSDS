\name{KS}
\alias{KS}
\alias{KS.deriv}

\title{Kalman Smoother for State Space Models}

\description{
These functions run the iterative equations of the Kalman smoother 
for a state space model upon the output from the Kalman filter.
}

\usage{
KS(y, ss, kf)
KS.deriv(y, ss, kf)
}

\arguments{
\item{y}{a numeric time series or vector.}
\item{ss}{a list containing the matrices of the state space model.}
\item{kf}{a list containing the output returned by the Kalman filter 
\code{\link[KFKSDS]{KF}}.}
}

\details{
See the details section and the section \sQuote{state space representation}
in \code{\link[KFKSDS]{KF}}.

Missing observations are allowed.

The input \code{kf} passed to \code{KS.deriv} must contain the 
derivative terms related to the filter that are returned by
\code{\link[KFKSDS]{KF.deriv}} or \code{\link[KFKSDS]{KF.deriv.C}}.

When the Kalman filter was found to convergence at some iteration, 
i.e., \code{kf$convit} is not null, these functions use steady state values for
\code{N} and \code{varahat} in the intermediate iterations of the smoother.
For example, if the filter converged at iteration \eqn{15} in a series of length \eqn{n}, 
the equations of the smoother are run for the first iterations from observation \eqn{n} 
to \eqn{n-15}; then the steady state values are used until there are \eqn{15} iterations 
remaining. In the last iterations, from observation \eqn{15} to \eqn{1} the equations of 
the smoother are evaluated again.

In practice, if the disturbance smoother is to be run as well, using the functions 
described in \code{\link[KFKSDS]{KFKSDS}} will be slightly more efficient.
}

\value{
A list containing the following elements:
\item{ahat}{smoothed state vector.}
\item{varhat}{covariance matrix of \code{ahat}.}
\item{r}{weighted sum of innovations used to obtain \code{ahat}.}
\item{N}{intermediate matrix used to obtain \code{varahat}}

The function \code{KS.deriv} returns also the derivatives referred to each 
of the elements defined above, named respectively \code{dahat}, \code{dvarahat}, 
\code{dr} and \code{dN}.
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
\code{\link[KFKSDS]{KF}}, \code{\link[KFKSDS]{KSDS}};
\code{\link[stsm.class]{char2numeric}} in package \pkg{stsm.class}.
}

\examples{
# local level plus seasonal model with arbitrary parameter values
# for the 'JohnsonJohnson' time series
m <- stsm.class::stsm.model(model = "llm+seas", y = JohnsonJohnson, 
  pars = c("var1" = 2, "var2" = 15, "var3" = 30))
ss <- stsm.class::char2numeric(m)

kf <- KF(m@y, ss)
ks <- KS(m@y, ss, kf)

plot(ks$ahat[,1:2], main = "smoothed state vector")

kfd <- KF.deriv(m@y, ss)
ksd <- KS.deriv(m@y, ss, kfd)
all.equal(ks$ahat, ksd$ahat)

# extended output is required if 'KF.deriv.C' is used to obtain 
# the necessary elements from the filter, set return.all = TRUE
kfdc <- KF.deriv.C(m@y, ss, return.all = TRUE)
ksd <- KS.deriv(m@y, ss, kfdc)
all.equal(ks$ahat, ksd$ahat)

# compare analytical and numerical derivatives
# yield same results up to a tolerance error
fcn <- function(x, model, type, i)
{
  m <- stsm.class::set.pars(model, x)
  ss <- stsm.class::char2numeric(m)
  kf <- KF(m@y, ss)
  ks <- KS(m@y, ss, kf)
  switch(type, "ahat" = sum(ks$ahat[,i]), "r" = sum(ks$r[,i]))
}

dahat <- numDeriv::grad(func = fcn, x = m@pars, model = m, type = "ahat", i = 1)
all.equal(dahat, colSums(ksd$dahat[,1,]))
dahat <- numDeriv::grad(func = fcn, x = m@pars, model = m, type = "ahat", i = 2)
all.equal(dahat, colSums(ksd$dahat[,2,]))
dahat <- numDeriv::grad(func = fcn, x = m@pars, model = m, type = "ahat", i = 3)
all.equal(dahat, colSums(ksd$dahat[,3,]))
dr <- numDeriv::grad(func = fcn, x = m@pars, model = m, type = "r", i = 1)
all.equal(dr, colSums(ksd$dr[,1,]), check.attributes = FALSE)
dr <- numDeriv::grad(func = fcn, x = m@pars, model = m, type = "r", i = 2)
all.equal(dr, colSums(ksd$dr[,2,]), check.attributes = FALSE)
dr <- numDeriv::grad(func = fcn, x = m@pars, model = m, type = "r", i = 3)
all.equal(dr, colSums(ksd$dr[,3,]), check.attributes = FALSE)
}

\keyword{ts, model}
